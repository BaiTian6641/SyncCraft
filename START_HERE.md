# Start here: hand SyncCraft to your local agent

## 1. Open the extracted folder

Open `synccraft-starter/` as the project root in Antigravity or your other local
agent software. Do not feed only the PDF: this folder contains the executable
baseline, exact boundaries, task queue, and verification evidence.

## 2. Give the agent the handoff prompt

Copy the complete contents of `agent/LOCAL_AGENT_PROMPT.md` into a new agent task.
It tells the agent to inspect the actual repository, reproduce the CPU baseline,
complete local preflight and import checks, and then work through Phase 1 in
dependency order. `AGENTS.md` contains the persistent engineering rules.

The first meaningful deliverable is a version-pinned, engine-imported static
vertical slice. The later deliverable is the complete evaluated Repair/PGD pipeline.
The current starter is neither of those completed milestones.

## 3. Reproduce the baseline

```bash
python scripts/check.py
python scripts/demo.py
```

Use the setup commands in `README.md` if NumPy or Pillow is missing. The demos
write `runs/demo/`. The delivered copies under `artifacts/examples/` let you inspect
the outputs immediately. In each example, open `preview.png`, then `asset.bbmodel`
in your installed Blockbench and inspect the hierarchy, pivots, face mapping, and
mirroring. Record the actual Blockbench version and save/reopen result.

## 4. Keep the important boundaries visible

- This release has static **box** support. It refuses the report's full extrusion,
  shading, grip, and animation example rather than dropping those statements.
- “Audit passed” currently means the listed starter geometry checks passed. It
  does not certify production usability, hand fit, zero skating, or runtime import.
- Model selection is open. Three candidate labels are carried from the report;
  exact repositories, revisions, licenses, tokenizer behavior, and GPU memory must
  be verified locally. The starter downloads no model and launches no paid job.
- The learned refiner remains behind G2. Missing evidence keeps that gate blocked.
  If the preregistered Repair/PGD policy reaches at least 90%, skip that branch.

## 5. Where the agent resumes

Read these in order:

1. `AGENTS.md` and `docs/STATUS.md`.
2. `docs/PLAN_RECONCILIATION.md` and `docs/adr/0001-starter-scope.md`.
3. `docs/ARCHITECTURE.md`, `docs/DSL_SUBSET.md`, and `docs/GEOMETRY_CONVENTIONS.md`.
4. `docs/tasks/BACKLOG.md` and `docs/ACCEPTANCE.md`.
5. The original report for the full research intent and later phases.

The PDF visibly cuts off the animation grammar on page 4 and refiner code on page
10. Those missing portions must be recovered or explicitly re-specified when their
tasks become active. They do not prevent the static-box starter from running.

## 中文交接摘要

将解压后的 `synccraft-starter` 文件夹作为本地智能体的工作目录，并把
`agent/LOCAL_AGENT_PROMPT.md` 的完整内容交给它。先复现现有测试，再按任务依赖完成
静态资产垂直切片。当前版本已包含可运行的 DSL、刚体变换、SAT 检查和导出基础；
不代表论文中的完整系统已经实现。模型训练、可微 PGD、动画、Bedrock/GeckoLib
以及学习型投影器仍有明确的待办和验收条件。G2 没有真实评测证据时保持阻塞。
