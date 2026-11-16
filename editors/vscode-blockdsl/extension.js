// @ts-check
"use strict";

const vscode = require("vscode");
const cp = require("child_process");
const fs = require("fs");
const path = require("path");

const { indexDocument } = require("./src/symbol_index");
const { provideCompletions } = require("./src/completion_engine");
const { provideHover, extractTokenAt } = require("./src/hover_docs");
const { runFastDiagnostics } = require("./src/diagnostics_engine");

const KIND_MAP = {
  Keyword: vscode.CompletionItemKind.Keyword,
  Snippet: vscode.CompletionItemKind.Snippet,
  Property: vscode.CompletionItemKind.Property,
  Field: vscode.CompletionItemKind.Field,
  Struct: vscode.CompletionItemKind.Struct,
  Function: vscode.CompletionItemKind.Function,
  Enum: vscode.CompletionItemKind.Enum,
  EnumMember: vscode.CompletionItemKind.EnumMember,
  Constant: vscode.CompletionItemKind.Constant,
  Color: vscode.CompletionItemKind.Color,
  File: vscode.CompletionItemKind.File,
  Interface: vscode.CompletionItemKind.Interface,
  Reference: vscode.CompletionItemKind.Reference
};

/**
 * Embedded zero-eval Python bridge script executed via `execFile(pythonBin, ["-c", ...])`,
 * passing the active `.blockdsl` source safely over `stdin` (never string-interpolated!).
 */
const PYTHON_AUDIT_SCRIPT = [
  "import json, re, sys",
  "src = sys.stdin.read()",
  "mode = sys.argv[1]",
  "check_clips = sys.argv[2] == '1'",
  "from synccraft.dsl.parser import parse_program, format_program",
  "if mode == 'format':",
  "    prog = parse_program(src)",
  "    sys.stdout.write(json.dumps({'ok': True, 'formatted': format_program(prog)}))",
  "    raise SystemExit(0)",
  "from synccraft.kinematics.se3_chain import compile_geometry",
  "from synccraft.audit.sync_audit import audit_scene",
  "from synccraft.kinematics.musculoskeletal import audit_behavior_clip",
  "try:",
  "    prog = parse_program(src)",
  "    scene = compile_geometry(prog)",
  "    aud = audit_scene(scene, enforce_physical_contact=True)",
  "    clip_issues = []",
  "    if check_clips and prog.behavior and prog.behavior.clips:",
  "        for clip in prog.behavior.clips:",
  "            ca = audit_behavior_clip(prog, clip.name, num_frames=12)",
  "            if not ca['passed']:",
  "                clip_issues.append({'clip': clip.name, 'collisions': ca['swept_collision_frame_count'], 'slip': ca['max_stance_slip_units']})",
  "    sys.stdout.write(json.dumps({",
  "        'ok': True,",
  "        'passed': bool(aud['passed']) and len(clip_issues) == 0,",
  "        'audit_profile': prog.meta.audit_profile,",
  "        'boxes': len(scene.boxes),",
  "        'bones': len(scene.bones),",
  "        'sculpted_overlaps': len(aud.get('sculpted_overlap_pairs', ())),",
  "        'collisions': aud['collision_pairs'][:12],",
  "        'contacts': aud['contact_violations'][:12],",
  "        'z_fights': aud['z_fighting_pairs'][:8],",
  "        'clip_issues': clip_issues",
  "    }))",
  "except Exception as exc:",
  "    msg = str(exc)",
  "    m = re.search(r'line\\s+(\\d+)(?:,\\s*col(?:umn)?\\s+(\\d+))?', msg, re.I)",
  "    line = int(m.group(1)) - 1 if m else 0",
  "    col = int(m.group(2)) - 1 if (m and m.group(2)) else 0",
  "    sys.stdout.write(json.dumps({'ok': False, 'error_type': type(exc).__name__, 'message': msg, 'line': max(0, line), 'col': max(0, col)}))"
].join("\n");

/**
 * Resolve Python interpreter + `PYTHONPATH` for the current workspace if available.
 * @param {vscode.TextDocument} document
 */
function resolveSynccraftPython(document) {
  const cfg = vscode.workspace.getConfiguration("synccraft.blockdsl", document.uri);
  const configured = (cfg.get("pythonPath") || "").trim();
  const folder = vscode.workspace.getWorkspaceFolder(document.uri);
  const roots = [];
  if (folder) roots.push(folder.uri.fsPath);
  const docDir = path.dirname(document.uri.fsPath);
  roots.push(docDir, path.resolve(docDir, ".."), path.resolve(docDir, "../.."));

  let pythonBin = configured || "python3";
  let repoRoot = null;
  for (const r of roots) {
    const venvUnix = path.join(r, ".venv", "bin", "python");
    const venvWin = path.join(r, ".venv", "Scripts", "python.exe");
    if (!configured && fs.existsSync(venvUnix)) {
      pythonBin = venvUnix;
    } else if (!configured && fs.existsSync(venvWin)) {
      pythonBin = venvWin;
    }
    if (!repoRoot && fs.existsSync(path.join(r, "src", "synccraft", "__init__.py"))) {
      repoRoot = r;
    }
  }
  const env = { ...process.env };
  if (repoRoot) {
    const srcPath = path.join(repoRoot, "src");
    env.PYTHONPATH = env.PYTHONPATH ? `${srcPath}${path.delimiter}${env.PYTHONPATH}` : srcPath;
  }
  return { pythonBin, env };
}

/**
 * Run the embedded SyncCraft Python bridge (`audit` or `format`).
 * @param {vscode.TextDocument} document
 * @param {"audit" | "format"} mode
 * @param {boolean} checkClips
 * @returns {Promise<any | null>}
 */
function invokePythonBridge(document, mode, checkClips) {
  const { pythonBin, env } = resolveSynccraftPython(document);
  return new Promise((resolve) => {
    const child = cp.execFile(
      pythonBin,
      ["-c", PYTHON_AUDIT_SCRIPT, mode, checkClips ? "1" : "0"],
      { env, timeout: 15000, maxBuffer: 4 * 1024 * 1024 },
      (err, stdout) => {
        if (err || !stdout) {
          resolve(null);
          return;
        }
        try {
          resolve(JSON.parse(stdout));
        } catch {
          resolve(null);
        }
      }
    );
    if (child.stdin) {
      child.stdin.write(document.getText(), "utf8");
      child.stdin.end();
    }
  });
}

/**
 * @param {vscode.ExtensionContext} context
 */
function activate(context) {
  const selector = [{ language: "blockdsl" }];
  const fastDiagCollection = vscode.languages.createDiagnosticCollection("blockdsl-fast");
  const compilerDiagCollection = vscode.languages.createDiagnosticCollection("blockdsl-audit");
  const statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  statusBar.command = "synccraft.blockdsl.auditActiveFile";
  context.subscriptions.push(fastDiagCollection, compilerDiagCollection, statusBar);

  /** Cache latest document index by URI + version for sub-millisecond completion/hover/outline */
  const cache = new Map();
  function getDocIndex(document) {
    const key = document.uri.toString();
    const hit = cache.get(key);
    if (hit && hit.version === document.version) {
      return hit.index;
    }
    const index = indexDocument(document.getText());
    cache.set(key, { version: document.version, index });
    return index;
  }

  function updateFastDiagnostics(document) {
    if (document.languageId !== "blockdsl") return;
    const idx = getDocIndex(document);
    const raw = runFastDiagnostics(idx);
    const vDiags = raw.map((d) => {
      const range = new vscode.Range(d.start.line, d.start.character, d.end.line, d.end.character);
      const diag = new vscode.Diagnostic(
        range,
        d.message,
        d.severity === "error" ? vscode.DiagnosticSeverity.Error : vscode.DiagnosticSeverity.Warning
      );
      diag.source = "blockdsl-lint";
      diag.code = d.code;
      return diag;
    });
    fastDiagCollection.set(document.uri, vDiags);
  }

  async function runDeepCompilerAudit(document, showToast) {
    if (!document || document.languageId !== "blockdsl") return;
    const cfg = vscode.workspace.getConfiguration("synccraft.blockdsl", document.uri);
    if (!cfg.get("enableCompilerAudit", true) && !showToast) return;

    statusBar.text = "$(sync~spin) Block-DSL: Auditing...";
    statusBar.show();

    const checkClips = cfg.get("auditBehaviorClipsOnSave", true);
    const res = await invokePythonBridge(document, "audit", checkClips);
    if (!res) {
      statusBar.text = "$(info) Block-DSL: Lint Active";
      if (showToast) {
        vscode.window.showWarningMessage(
          "SyncCraft Python package not found. Configure `synccraft.blockdsl.pythonPath` to `.venv/bin/python` for full SE(3) + 15-axis SAT audits."
        );
      }
      return;
    }

    const idx = getDocIndex(document);
    const diags = [];
    const findSymbolRange = (symName) => {
      const baseName = symName.includes("__") ? symName.split("__").pop() : symName;
      const hit =
        idx.symbols.expandedBoxes.find((b) => b.name === symName || b.name === baseName) ||
        idx.symbols.expandedBones.find((b) => b.name === symName || b.name === baseName);
      if (hit && hit.position) {
        return new vscode.Range(hit.position.line, hit.position.character, hit.position.line, hit.position.character + baseName.length);
      }
      return new vscode.Range(0, 0, 0, 5);
    };

    if (!res.ok) {
      diags.push(
        new vscode.Diagnostic(
          new vscode.Range(res.line, res.col, res.line, res.col + 16),
          `${res.error_type}: ${res.message}`,
          vscode.DiagnosticSeverity.Error
        )
      );
      statusBar.text = "$(error) Block-DSL: Parse/Validation Error";
    } else {
      for (const col of res.collisions || []) {
        const d = new vscode.Diagnostic(
          findSymbolRange(col.a),
          `15-Axis SAT Collision (${res.audit_profile}): '${col.a}' overlaps '${col.b}' by ${Number(col.depth).toFixed(4)}u`,
          vscode.DiagnosticSeverity.Error
        );
        d.source = "synccraft-sat";
        diags.push(d);
      }
      for (const cv of res.contacts || []) {
        const targetName = cv.a || cv.component_Representative || "";
        const d = new vscode.Diagnostic(
          findSymbolRange(targetName),
          `Contact Contract Violation (${cv.kind}): ${JSON.stringify(cv)}`,
          vscode.DiagnosticSeverity.Error
        );
        d.source = "synccraft-contact";
        diags.push(d);
      }
      for (const zf of res.z_fights || []) {
        const d = new vscode.Diagnostic(
          findSymbolRange(zf.a),
          `Coplanar Z-Fighting (${zf.kind}): '${zf.a}' vs '${zf.b}'`,
          vscode.DiagnosticSeverity.Error
        );
        d.source = "synccraft-zfight";
        diags.push(d);
      }
      for (const ci of res.clip_issues || []) {
        const clipSym = idx.symbols.clips.find((c) => c.name === ci.clip);
        const r = clipSym
          ? new vscode.Range(clipSym.position.line, clipSym.position.character, clipSym.position.line, clipSym.position.character + clipSym.name.length)
          : new vscode.Range(0, 0, 0, 5);
        const d = new vscode.Diagnostic(
          r,
          `Swept @behavior clip '${ci.clip}' failed: collisions=${ci.collisions}, max_stance_slip=${Number(ci.slip).toFixed(4)}u`,
          vscode.DiagnosticSeverity.Error
        );
        d.source = "synccraft-behavior";
        diags.push(d);
      }

      if (res.passed) {
        const scNote = res.sculpted_overlaps > 0 ? `, ${res.sculpted_overlaps} sculpted` : "";
        statusBar.text = `$(pass) Block-DSL (${res.audit_profile}): ${res.boxes} boxes, ${res.bones} bones${scNote}`;
        if (showToast) {
          vscode.window.showInformationMessage(
            `Block-DSL Audit Passed (${res.audit_profile}): ${res.boxes} boxes, ${res.bones} bones${scNote}, 0 collisions.`
          );
        }
      } else {
        statusBar.text = `$(error) Block-DSL: ${diags.length} Audit Issue(s)`;
        if (showToast) {
          vscode.window.showErrorMessage(`Block-DSL Audit found ${diags.length} collision/contact/clip issue(s).`);
        }
      }
    }
    compilerDiagCollection.set(document.uri, diags);
  }

  // 1. Completion Provider
  context.subscriptions.push(
    vscode.languages.registerCompletionItemProvider(
      selector,
      {
        provideCompletionItems(document, position) {
          const idx = getDocIndex(document);
          const offset = document.offsetAt(position);
          const rawItems = provideCompletions(idx, offset);
          return rawItems.map((it) => {
            const item = new vscode.CompletionItem(it.label, KIND_MAP[it.kind] || vscode.CompletionItemKind.Text);
            item.detail = it.detail;
            if (it.documentation) {
              item.documentation = new vscode.MarkdownString(it.documentation);
            }
            item.insertText = it.isSnippet ? new vscode.SnippetString(it.insertText) : it.insertText;
            item.sortText = it.sortText;
            return item;
          });
        }
      },
      "@",
      ":",
      "=",
      ".",
      "[",
      "(",
      " "
    )
  );

  // 2. Hover Provider
  context.subscriptions.push(
    vscode.languages.registerHoverProvider(selector, {
      provideHover(document, position) {
        const idx = getDocIndex(document);
        const offset = document.offsetAt(position);
        const res = provideHover(idx, offset);
        if (!res) return null;
        return new vscode.Hover(new vscode.MarkdownString(res.markdown));
      }
    })
  );

  // 3. Go-to-Definition Provider (jumps to bone, box, plane, locator, palette, macro, muscle, ik_chain, clip, state — even from `mirror_x` `twin__box` references!)
  context.subscriptions.push(
    vscode.languages.registerDefinitionProvider(selector, {
      provideDefinition(document, position) {
        const idx = getDocIndex(document);
        const tok = extractTokenAt(document.getText(), document.offsetAt(position));
        if (!tok) return null;
        const hit =
          idx.symbols.globalDeclarations.find((d) => d.name === tok) ||
          (idx.symbols.macros.has(tok) ? idx.symbols.macros.get(tok) : null) ||
          idx.symbols.lets.find((l) => l.name === tok);
        if (!hit || !hit.position) return null;
        const pos = new vscode.Position(hit.position.line, hit.position.character);
        return new vscode.Location(document.uri, pos);
      }
    })
  );

  // 4. Document Outline (Symbol) Provider
  context.subscriptions.push(
    vscode.languages.registerDocumentSymbolProvider(selector, {
      provideDocumentSymbols(document) {
        const idx = getDocIndex(document);
        const out = [];
        for (const sc of idx.scopes) {
          if (sc.kind === "section") {
            const start = document.positionAt(sc.start);
            const end = document.positionAt(sc.end);
            out.push(
              new vscode.DocumentSymbol(
                sc.name,
                "Section",
                vscode.SymbolKind.Namespace,
                new vscode.Range(start, end),
                new vscode.Range(start, start.translate(0, sc.name.length))
              )
            );
          } else if (sc.kind === "root_bone" || sc.kind === "bone" || sc.kind === "macro" || sc.kind === "clip" || sc.kind === "muscle") {
            const start = document.positionAt(sc.start);
            const end = document.positionAt(sc.end);
            const symKind =
              sc.kind === "macro" || sc.kind === "clip"
                ? vscode.SymbolKind.Function
                : (sc.kind === "muscle" ? vscode.SymbolKind.Event : vscode.SymbolKind.Struct);
            out.push(
              new vscode.DocumentSymbol(
                sc.name,
                sc.kind,
                symKind,
                new vscode.Range(start, end),
                new vscode.Range(start, start)
              )
            );
          }
        }
        return out;
      }
    })
  );

  // 5. Document Formatting Provider (Canonical `format_program`)
  async function computeFormattingEdits(document) {
    const res = await invokePythonBridge(document, "format", false);
    if (!res || !res.ok || typeof res.formatted !== "string") {
      return [];
    }
    const fullRange = new vscode.Range(
      document.positionAt(0),
      document.positionAt(document.getText().length)
    );
    return [vscode.TextEdit.replace(fullRange, res.formatted)];
  }

  context.subscriptions.push(
    vscode.languages.registerDocumentFormattingEditProvider(selector, {
      provideDocumentFormattingEdits(document) {
        return computeFormattingEdits(document);
      }
    })
  );

  // 6. Commands & Editor Lifecycle Listeners
  context.subscriptions.push(
    vscode.commands.registerCommand("synccraft.blockdsl.auditActiveFile", async () => {
      const editor = vscode.window.activeTextEditor;
      if (editor && editor.document.languageId === "blockdsl") {
        updateFastDiagnostics(editor.document);
        await runDeepCompilerAudit(editor.document, true);
      }
    }),
    vscode.commands.registerCommand("synccraft.blockdsl.formatActiveFile", async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor || editor.document.languageId !== "blockdsl") return;
      const edits = await computeFormattingEdits(editor.document);
      if (edits.length > 0) {
        const wsEdit = new vscode.WorkspaceEdit();
        wsEdit.set(editor.document.uri, edits);
        await vscode.workspace.applyEdit(wsEdit);
      } else {
        vscode.window.showWarningMessage("Cannot format Block-DSL: fix syntax errors or configure `synccraft.blockdsl.pythonPath`.");
      }
    }),
    vscode.workspace.onDidChangeTextDocument((e) => {
      if (e.document.languageId === "blockdsl") {
        updateFastDiagnostics(e.document);
      }
    }),
    vscode.workspace.onDidSaveTextDocument((doc) => {
      if (doc.languageId === "blockdsl") {
        updateFastDiagnostics(doc);
        runDeepCompilerAudit(doc, false);
      }
    }),
    vscode.workspace.onDidOpenTextDocument((doc) => {
      if (doc.languageId === "blockdsl") {
        updateFastDiagnostics(doc);
        runDeepCompilerAudit(doc, false);
      }
    }),
    vscode.window.onDidChangeActiveTextEditor((ed) => {
      if (ed && ed.document.languageId === "blockdsl") {
        statusBar.show();
        updateFastDiagnostics(ed.document);
      } else {
        statusBar.hide();
      }
    })
  );

  if (vscode.window.activeTextEditor && vscode.window.activeTextEditor.document.languageId === "blockdsl") {
    const doc = vscode.window.activeTextEditor.document;
    updateFastDiagnostics(doc);
    runDeepCompilerAudit(doc, false);
  }
}

function deactivate() {}

module.exports = {
  activate,
  deactivate
};
