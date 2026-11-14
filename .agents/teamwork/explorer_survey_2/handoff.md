# Handoff Report: SyncCraft Dataset Pipeline, Collision Audit, and Manifest Survey

## 1. Observation

### 1.1 Dataset Scripts and Data Policy Status
- **`src/synccraft/data_gen/static_families.py:1-8`**:
  ```python
  """Dataset scale-up remains blocked on full static audit and versioned target checks."""
  def generate_dataset(*args, **kwargs):
      raise NotImplementedError(
          "P1-08: implement a reviewed 200-example pilot with provenance, frozen "
          "family splits, complete audit labels and deterministic shards before scale-up."
      )
  ```
- **`src/synccraft/data_gen/ik_motion_templates.py:1-8`**:
  ```python
  """Motion is Phase 3; the full grammar must first be recovered or specified."""
  def generate_ik_clip(*args, **kwargs):
      raise NotImplementedError(
          "P3-01: implement stance-foot world locks, reachability handling and "
          "parameterized motion fits, then measure slip and exported-curve error."
      )
  ```
- **`configs/data_policy.yaml:1-52`**:
  - `schema: synccraft.data-policy/0.1`
  - `status: specification_no_dataset_bundled`
  - `source_plan_scale_targets`: `static_train: 60000`, `dynamic_train: 15000`, `synthetic_unseen_total: 1200`, `external_gold_total: 50`
  - `pilot_samples_before_scale: 200`, `seeds: [17, 29, 43]`
  - `split_by: family_and_body_plan_before_parameter_sampling`
  - `seen_static_families`: `straight_blade_sword`, `single_edge_blade`, `battle_axe`, `spear`, `war_hammer`, `simple_wood_bow`, `furniture_table_chair_chest_cabinet`, `gearbox`
  - `unseen_static_families`: `mechanical_crossbow`, `hand_lantern`, `tripod_telescope`, `folding_windmill`, `anvil_forge`, `floating_ring_staff`
  - `seen_body_plans`: `humanoid_biped`, `dog_horse_quadruped`, `bird`
  - `unseen_body_plans`: `hexapod_beetle`, `heavy_tail_tyrannosaur`, `segmented_sea_dragon`
  - `prohibited_training_sources`: `original_mojang_models_and_textures`, `third_party_ip_derivatives_without_suitable_rights`, `incompatible_or_unknown_motion_dataset_licenses`, `heldout_g2_or_external_benchmark_assets`
  - `required_row_fields`: `[asset_id, family_id, body_plan_id_if_dynamic, generator_revision, generator_parameters, canonical_dsl, canonical_dsl_sha256, prompt_and_provenance, source_license_and_review_record, split, audit_version_and_results, target_profile_and_version, seed]`
  - `shard_target_megabytes: 500`, `shard_format: choose_after_pilot_parquet_or_arrow`

### 1.2 Intermediate Representation & Asset Processing Chain
- **Parser (`src/synccraft/dsl/parser.py`)**:
  - `_tokens(text)` parses custom Block-DSL tokens (`_LEXER` matches comments, numbers, identifiers, punctuation). Input size limited to 2 MB and 200,000 tokens (`lines 37-56`).
  - `_Parser.program()` parses sections `@meta`, `@palette`, and `@rig root bone ...` into typed immutable dataclasses without using `eval` or dynamic execution.
- **AST and Semantic Validation (`src/synccraft/dsl/ast_nodes.py`)**:
  - `validate_program(program)` (`lines 119-247`):
    - Validates metadata (`archetype` in `{"entity", "weapon_3d", "block_multipart", "item_extruded"}`, `target_profile` in `{"bedrock_entity", "geckolib", "java_modern", "java_legacy", "bbmodel_5"}`, `texel_density > 0`).
    - Enforces entity vs Java incompatibility (`lines 132-133`: `if meta.archetype == "entity" and meta.target_profile.startswith("java_"): raise DSLValidationError(...)`).
    - Rejects `@shade` and `@ergonomics` as unimplemented in starter (`lines 134-135`).
    - Validates palettes (`OKLCH` base lightness in `[0,1]`, chroma $\ge 0$, steps in `[2,256]`, emissive boolean).
    - Traverses bone tree (max depth 64, max declarations 4096).
    - Child bone must have `Attach` referencing preceding box in direct parent bone; `anchor` and `align` components in `[-1, 1]`; connected attach `offset` in `[-0.5, 0.5]` (`lines 190-201`).
    - Boxes must have 3 positive integer dimensions, valid material, `inflate >= 0`, valid unique `cullface` names.
    - Subsequent boxes in the same bone require `place` referencing an earlier box in the same bone (`lines 226-240`).
- **Forward Kinematics (`src/synccraft/kinematics/se3_chain.py`)**:
  - `compile_geometry(program) -> Scene`:
    - Evaluates Euler angles via column-vector convention `Rz @ Ry @ Rx` (`euler_xyz_matrix`, `lines 26-34`).
    - Root bone placed at `pivot_local`.
    - Child bone translation: `attachment_base.local_center + 0.5 * attachment_base.size * anchor * sign + offset * sign` (`lines 181-185`).
    - World pivot: `parent.pivot_world + parent.rotation @ translation` (`line 186`).
    - Primary box of child bone attaches to parent box: `edge(attachment_base.name, box_name, "attach")` if `not bone.attach.detached` (`lines 230-231`).
    - Subsequent boxes placed relative to base box: `edge(local_boxes[item.place.base_box].name, box_name, "place")` (`lines 232-233`).
    - `mirror_x(mirror_name)` reflects X coordinate (`reflect = (-1, 1, 1)`, `angle_reflect = (1, -1, -1)`), recursively creating sibling bones/boxes with prefixed names (`lines 234-244`).
    - Output: `Scene(boxes, bones, adjacency, meta, adjacency_reasons)`.
- **Exporters and Rendering**:
  - `src/synccraft/shading/uv_packer.py`: `build_atlas(program, scene)` packs integer UV islands.
  - `src/synccraft/backends/bbmodel5_exporter.py`: `export_bbmodel(scene, name, atlas)` produces Blockbench 5.0 JSON.
  - `src/synccraft/backends/java_exporter.py`: `export_java(scene, profile, atlas)` produces Java block/item models.
  - `src/synccraft/audit/offscreen_renderer.py`: `render_four_views(program, scene, output_path)` produces 1200x920 4-view preview PNG.

### 1.3 Collision Audit & Geometry Validation
- **Direct SAT Collision (`src/synccraft/kinematics/sat_potential.py:51-95`)**:
  - `sat_obb(left, right, collision_tolerance=1e-7, beta=25.0, energy_tolerance=0.01, cross_axis_tolerance=1e-10)`:
    - Verifies box orthonormal rotation (`R.T @ R == I`) and proper determinant (`det(R) == 1`).
    - Tests up to 15 axes: 3 from `left.rotation`, 3 from `right.rotation`, 9 cross products.
    - **Degenerate axis masking**:
      ```python
      candidate = np.cross(left.rotation[:, i], right.rotation[:, j])
      norm = np.linalg.norm(candidate)
      if norm > cross_axis_tolerance:
          axes.append(candidate / norm)
      ```
      Axes with `norm <= 1e-10` are excluded from testing.
    - Projections: `radii_left = np.abs(axis_array @ left.rotation) @ left.half_extents`, `radii_right = np.abs(axis_array @ right.rotation) @ right.half_extents`.
    - Distance: `distance = np.abs(axis_array @ (right.center - left.center))`.
    - Overlaps: `overlaps = radii_left + radii_right - distance`.
    - Overlap decision: `raw_min = float(overlaps.min())`.
    - Collision flag: `collision = raw_min > collision_tolerance` (`1e-7`).
    - Soft minimum and surrogate energy:
      ```python
      soft_min = raw_min - float(np.log(np.exp(-beta * (overlaps - raw_min)).sum())) / beta
      penetration = max(raw_min, 0.0) if collision else 0.0
      energy = max(soft_min - energy_tolerance, 0.0) ** 2 if collision else 0.0
      ```
- **Audit Scene Filter (`src/synccraft/audit/sync_audit.py:34-97`)**:
  - Iterates over all unique 2-box combinations:
    ```python
    for left, right in combinations(scene.boxes, 2):
        pair = tuple(sorted((left.name, right.name)))
        result = sat_obb(left, right, collision_tolerance=collision_tolerance)
        entry = {"pair": list(pair), **result.to_dict()}
        if pair in scene.adjacency:
            entry["reason"] = scene.adjacency_reasons[pair]
            declared_contacts.append(entry)
        else:
            evaluated_pairs += 1
            if result.collision:
                collisions.append(entry)
    ```
  - `passed = not collisions and not errors`.
  - Disconnected component analysis via `_components(scene)` (`lines 12-31`).
  - Output report dictionary contains: `status`, `passed`, `box_count`, `bone_count`, `evaluated_non_exempt_pairs`, `collision_pairs`, `declared_contact_pairs`, `connected_components`, `errors`, `coverage_gaps`.

### 1.4 Dataset Manifests and Training/Gate Consumption
- **Manifest References**:
  - `configs/training_static.yaml:5, 36`:
    - `dataset_manifest: null`
    - `checkpoint.include`: `[adapters, refiner_if_enabled, optimizer, scheduler, scaler_if_used, rng_all_devices, sampler, shard_cursor, global_step, config_hash, model_revision, dataset_manifest]`
  - `configs/gate_g2_evidence.json:5`:
    - `"dataset_manifest_sha256": null`
  - `src/synccraft/evaluation/gate_g2.py:16-18`:
    - Validates that `dataset_manifest_sha256` and `evaluation_manifest_sha256` match `re.fullmatch(r"[a-fA-F0-9]{64}", ...)`.

---

## 2. Logic Chain

1. **Procedural Generation Foundation vs Starter Stubs (From Obs 1.1)**:
   - While `static_families.py` raises `NotImplementedError`, the full specification is present in `configs/data_policy.yaml`, and the underlying geometry compilation (`compile_geometry`), AST validation (`validate_program`), and audit pipeline (`audit_scene`) are fully implemented and passing in Linux validation (`artifacts/validation/summary.json`).
   - Therefore, a dataset generation stage in the notebook can synthesize or gather raw Block-DSL programs (or read bundled examples/pilots), pass them through `parse_program`, `validate_program`, `compile_geometry`, and `audit_scene`, and store only the passing assets.

2. **Rigorous Collision Filtering via Direct SAT (From Obs 1.3)**:
   - As proven by `tests/test_sat.py` and `docs/NUMERICAL_NOTES.md`, the smooth surrogate potential (`softmin`) can yield 0 energy on true 0.01-unit overlaps (such as identical boxes offset to `[1.99, 1.99, 1.99]`).
   - Hence, the dataset filtering pipeline must rely strictly on `sat_obb(...).collision` / `raw_min > collision_tolerance` (direct SAT) rather than optimization surrogate energies.
   - Degenerate cross axes must be masked out when their norm $\le 10^{-10}$ to prevent false-zero minimum overlaps.

3. **Strict Contact Exclusion Contract (From Obs 1.2 & 1.3)**:
   - Under `compile_geometry`, only two types of adjacency edges are generated:
     a. `attach` between attachment base box and primary box of child bone (if `detached=False`).
     b. `place` between a box and its explicit `base_box` within the same bone.
   - Sibling boxes and non-declared attachments have NO blanket immunity.
   - In `audit_scene`, declared contacts are stored in `declared_contact_pairs` and exempted from failing the audit. Any penetration in `evaluated_non_exempt_pairs` triggers `passed = False` and populates `collision_pairs`.
   - Therefore, any candidate DSL that results in non-exempt collisions is rejected by the dataset filter.

4. **Manifest Architecture and Frozen Held-Out Splits (From Obs 1.1 & 1.4)**:
   - Data policy requires strict separation: seen families are routed to `train` and `val`, while unseen families (`mechanical_crossbow`, `hand_lantern`, `tripod_telescope`, `folding_windmill`, `anvil_forge`, `floating_ring_staff`) and unseen body plans are routed to `test_unseen`.
   - Row format requires the 13 specified fields including `canonical_dsl_sha256`, `split`, and full `audit_version_and_results`.
   - The top-level manifest file must compute and store the 64-hex-character `dataset_manifest_sha256`, which directly feeds into `configs/training_static.yaml` for checkpoint reproducibility and `configs/gate_g2_evidence.json` for Gate G2 validation.

5. **Colab Notebook Execution Modes (From Obs 1.1, 1.4, & System Requirements)**:
   - In **Demo / Headless Mode**:
     - The notebook executes without downloading large shards or requiring external network calls.
     - It gathers the built-in examples (`simple_blade`, `mirrored_wings`, `stacked_block`) and generates a small procedural pilot of seen and unseen families.
     - It feeds them through `parse_program` $\to$ `validate_program` $\to$ `compile_geometry` $\to$ `audit_scene`.
     - It tests an intentional collision fixture to verify that collision filtering works properly.
     - It outputs `train.jsonl`, `val.jsonl`, `test_unseen.jsonl`, and a verified `dataset_manifest.json` with valid SHA-256 hashes.
   - In **Full Scale Mode**:
     - Authenticates to Hugging Face Hub using `HF_TOKEN` from `google.colab.userdata.get('HF_TOKEN')` (or `os.environ`).
     - Downloads pre-generated large shards (up to 500 MB Parquet/Arrow) or initiates scaled procedural generation.
     - Verifies checksums, runs collision audits, outputs the full dataset manifest, and optionally publishes verified artifacts to the user's Hugging Face repository.

---

## 3. Caveats

1. **Host Environment Architecture**: The local host is macOS with Python 3.9.6, lacking `numpy` and `Pillow`. `typing.TypeAlias` in `src/synccraft/dsl/ast_nodes.py:8` requires Python $\ge 3.10$. However, on the Colab target environment (Python 3.11/3.12 with GPU), all dependencies (`numpy>=1.26`, `Pillow>=10`, `torch>=2.5`) are fully satisfied.
2. **Current Audit Scope Limits**: `audit_scene` certifies absence of non-exempt collisions and finite valid OBBs, but as noted in `docs/NUMERICAL_NOTES.md`, it does not yet perform full surface-contact mesh checks or Steve/Alex hand grip collision checking (task P1-02/P1-04).
3. **800 CC0 Icon Inventory**: As specified in `configs/data_policy.yaml:50-51`, the 800 reviewed CC0 icon collection is a planning ambition and is not bundled in the repo.

---

## 4. Conclusion

1. **Asset Processing Pipeline**: Fully established in pure Python/NumPy:
   - Text $\to$ `parse_program` $\to$ `validate_program` $\to$ `compile_geometry` $\to$ `Scene`.
   - Exporters (`bbmodel5_exporter`, `java_exporter`), atlas builder (`uv_packer`), and diagnostic 4-view renderer (`offscreen_renderer`) are functional.
2. **Collision Audit Engine**:
   - `sat_obb` evaluates 15 axes using direct SAT with `raw_min > 1e-7` detection.
   - Degenerate cross axes ($\|\mathbf{u}_i \times \mathbf{u}_j\| \le 10^{-10}$) are excluded.
   - Declared contacts (`attach`, `place`) in `scene.adjacency` are exempted, while non-exempt collisions cause audit failure.
3. **Dataset Manifest Contract**:
   - Manifests must adhere to schema `synccraft.data-policy/0.1` and `synccraft.dataset-manifest/0.1`.
   - Every row contains the 13 mandatory fields with frozen family splits and audit dictionaries.
   - The manifest SHA-256 digest (`dataset_manifest_sha256`) is required by `training_static.yaml` checkpoints and Gate G2 evaluation.
4. **Notebook Dataset Stage Architecture**:
   - Structured into clean callable stages:
     - `setup_environment(mode)`
     - `gather_raw_assets(mode, hf_token=None)`
     - `filter_and_audit_assets(raw_candidates)`
     - `build_dataset_manifest(passing_records, output_dir)`
   - Dual-mode support enables instant, error-free headless execution during automated tests (`pytest` / `nbconvert`) while scaling to full Colab G4 runs with Hugging Face integration.

---

## 5. Verification Method

To verify these findings independently:
1. **Inspect Code and Data Policies**:
   ```bash
   cat configs/data_policy.yaml
   cat configs/training_static.yaml
   cat configs/gate_g2_evidence.json
   ```
2. **Inspect SAT and Audit Implementation**:
   ```bash
   sed -n '51,95p' src/synccraft/kinematics/sat_potential.py
   sed -n '34,97p' src/synccraft/audit/sync_audit.py
   ```
3. **Review Implemented Tests (Reference execution log)**:
   - Check `artifacts/validation/cpu-tests.txt` lines 1-120. All 10 tests including `SATTests` (1,000 rotated flush contact cases, degenerate axis handling, softmin counterexample, declared exemptions) passed with 0 failures under Python 3.12.
4. **Invalidation Conditions**:
   - If `audit_scene` were modified to use `softmin_overlap` instead of `raw_min_overlap`, the audit would falsely accept corner-overlapping boxes.
   - If degenerate cross-axis filtering were removed from `sat_obb`, parallel and identical boxes would fail to report collisions.
   - If heldout unseen families were included in the training split, Gate G2 validity would be invalidated due to train/test contamination.
