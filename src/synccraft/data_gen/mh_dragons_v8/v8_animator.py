"""src/synccraft/data_gen/mh_dragons_v8/v8_animator.py

Vigorous 12-Clip Monster Hunter Action & Musculoskeletal FK Choreography Engine.

Authors 12 smooth, bold, game-authentic explicit FK animation clips per monster
(264 total clips across the 22-monster roster) with:
  - Heavy Monster Hunter anticipation / load-up, explosive strike snap, overshoot,
    and delayed secondary follow-through across segmented wing banks, wrist chains,
    multi-bone tails, cobra hoods, Frenzy feelers, and dorsal manifolds.
  - Multi-sample smooth trajectory baking (`24` samples per cyclic clip, dense
    hermite-ready keyframe curves for one-shot attacks) so exported `.bbmodel` and
    Bedrock `.animation.json` play identically with zero runtime IK requirement.
"""

from __future__ import annotations

import math
import re
from typing import Any, Sequence


def _num_suffix(name: str) -> int:
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else 0


def build_monster_12_clips(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Synthesize 12 vigorous, species-authentic Monster Hunter FK clips for `doc`."""
    bones = {b["name"]: b for b in doc["bones"]}
    bnames = set(bones.keys())
    mid = str(doc["id"]).lower()
    sclass = str(doc.get("species_class", "")).lower()

    clips: dict[str, dict[str, Any]] = {}

    def new_clip(name: str, duration: float, loop: bool = False, description: str = "") -> dict[str, Any]:
        c = {"duration": round(float(duration), 4), "loop": bool(loop), "description": description or name, "tracks": []}
        clips[name] = c
        return c

    def set_track(c: dict[str, Any], bone: str, keys: list[list[Any]], channel: str = "rotation") -> None:
        if bone not in bnames:
            return
        c["tracks"] = [t for t in c["tracks"] if not (t["bone"] == bone and t["channel"] == channel)]
        c["tracks"].append({"bone": bone, "channel": channel, "keys": keys})

    def add_curve(
        c: dict[str, Any],
        bone: str,
        times_norm: Sequence[float],
        vec3_list: Sequence[Sequence[float]],
        channel: str = "rotation",
    ) -> None:
        if bone not in bnames:
            return
        dur = float(c["duration"])
        # Resample through Catmull-Rom at ~18fps so linear Blockbench/Bedrock playback is buttery smooth!
        raw_t = [float(t) * dur for t in times_norm]
        raw_v = [[float(x) for x in v] for v in vec3_list]
        n_steps = max(len(raw_t), int(round(dur * 16.0)))
        dense_keys: list[list[Any]] = []
        for si in range(n_steps + 1):
            ts = round(dur * (si / n_steps), 4)
            # Catmull-Rom eval
            if ts <= raw_t[0]:
                val = raw_v[0]
            elif ts >= raw_t[-1]:
                val = raw_v[-1]
            else:
                val = raw_v[-1]
                for k in range(len(raw_t) - 1):
                    if raw_t[k] <= ts <= raw_t[k + 1]:
                        u = (ts - raw_t[k]) / max(1e-9, raw_t[k + 1] - raw_t[k])
                        p0 = raw_v[max(0, k - 1)]
                        p1 = raw_v[k]
                        p2 = raw_v[k + 1]
                        p3 = raw_v[min(len(raw_v) - 1, k + 2)]
                        u2, u3 = u * u, u * u * u
                        val = [
                            round(
                                0.5
                                * (
                                    2.0 * p1[a]
                                    + (-p0[a] + p2[a]) * u
                                    + (2.0 * p0[a] - 5.0 * p1[a] + 4.0 * p2[a] - p3[a]) * u2
                                    + (-p0[a] + 3.0 * p1[a] - 3.0 * p2[a] + p3[a]) * u3
                                ),
                                4,
                            )
                            for a in range(3)
                        ]
                        break
            dense_keys.append([ts, [round(float(x), 4) for x in val]])
        set_track(c, bone, dense_keys, channel=channel)

    def add_wave(
        c: dict[str, Any],
        bone: str,
        terms: Sequence[tuple[int, float, float, float]],
        *,
        channel: str = "rotation",
        cycles: int = 1,
        samples: int = 24,
    ) -> None:
        """`terms`: sequence of `(axis_0_1_2, amplitude, phase_deg, bias)`."""
        if bone not in bnames:
            return
        dur = float(c["duration"])
        keys: list[list[Any]] = []
        for i in range(samples + 1):
            ang = 2.0 * math.pi * cycles * (i / samples)
            v = [0.0, 0.0, 0.0]
            for ax, amp, ph_deg, bias in terms:
                v[ax] += bias + amp * math.sin(ang + math.radians(ph_deg))
            keys.append([round(dur * (i / samples), 4), [round(x, 4) for x in v]])
        keys[-1][1] = list(keys[0][1])
        set_track(c, bone, keys, channel=channel)

    def side_sign(bone: str) -> float:
        return 1.0 if float(bones[bone]["pivot"][0]) >= 0.0 else -1.0

    # Discover anatomical bone groups dynamically
    tails = sorted([n for n in bnames if re.match(r"^tail_\d+$", n)], key=_num_suffix)
    necks = sorted([n for n in bnames if n == "neck" or re.match(r"^neck_\d+$", n)], key=_num_suffix)
    chest_b = "chest" if "chest" in bnames else ("thorax" if "thorax" in bnames else None)
    pelvis_b = "pelvis" if "pelvis" in bnames else ("body" if "body" in bnames else None)
    head_b = "head" if "head" in bnames else ("skull" if "skull" in bnames else ("head_cranium" if "head_cranium" in bnames else None))
    jaw_b = "jaw" if "jaw" in bnames else ("mandible" if "mandible" in bnames else None)

    wing_roots = [n for n in ("wing_l", "wing_r") if n in bnames]
    wing_fores = [n for n in ("wing_fore_l", "wing_fore_r") if n in bnames]
    wing_wrists = [n for n in ("wing_wrist_l", "wing_wrist_r") if n in bnames]
    wing_mids = [n for n in ("wing_mid_l", "wing_mid_r") if n in bnames]
    wing_dists = [n for n in ("wing_distal_l", "wing_distal_r") if n in bnames]

    chains_l = sorted([n for n in bnames if n.startswith("chain_l_")], key=_num_suffix)
    chains_r = sorted([n for n in bnames if n.startswith("chain_r_")], key=_num_suffix)
    feelers = [n for n in ("feeler_l", "feeler_r", "gore_feeler_l", "shagaru_horn_r") if n in bnames]
    hoods = [n for n in ("hood_l", "hood_r") if n in bnames]
    rail_horns = [n for n in ("rail_horn_l", "rail_horn_r") if n in bnames]
    capacitors = [n for n in bnames if any(k in n for k in ("cap_", "radiator_", "shocker_", "dorsal_vent"))]

    has_forelegs = ("fore_upper_l" in bnames) and ("fore_upper_r" in bnames)
    has_hindlegs = ("hind_upper_l" in bnames) and ("hind_upper_r" in bnames)
    is_leviathan = "leviathan" in sclass or any(k in mid for k in ("lagiacrus", "mizutsune", "jin_dahaad"))
    is_fanged = "fanged" in sclass or "zinogre" in mid
    is_biped_wyvern = (not has_forelegs) and has_hindlegs

    # Helper: traveling tail wave
    def apply_tail_wave(c: dict[str, Any], amp_deg: float, axis: int = 1, cycles: int = 1, phase0: float = 0.0, pitch_bias: float = 0.0) -> None:
        for idx, tb in enumerate(tails):
            grow = 1.0 + idx * 0.24
            terms = [(axis, amp_deg * grow, phase0 - idx * 36.0, 0.0)]
            if abs(pitch_bias) > 1e-4:
                terms.append((0, 2.5, phase0 - idx * 25.0, pitch_bias))
            add_wave(c, tb, terms, cycles=cycles)

    def apply_neck_wave(c: dict[str, Any], amp_deg: float, axis: int = 0, cycles: int = 1, phase0: float = 0.0, bias: float = 0.0) -> None:
        scale = 1.0 / max(1.0, len(necks) * 0.72)
        for idx, nb in enumerate(necks):
            add_wave(c, nb, [(axis, amp_deg * scale, phase0 - idx * 28.0, bias * scale)], cycles=cycles)

    def apply_neck_curve(c: dict[str, Any], ts: Sequence[float], x_vals: Sequence[float], y_vals: Sequence[float] | None = None, z_vals: Sequence[float] | None = None) -> None:
        scale = 1.0 / max(1.0, len(necks) * 0.72)
        yv = y_vals or [0.0] * len(ts)
        zv = z_vals or [0.0] * len(ts)
        for nb in necks:
            add_curve(c, nb, ts, [[x * scale, y * scale, z * scale] for x, y, z in zip(x_vals, yv, zv)])

    # =========================================================================
    # 1. IDLE (Vigorous living Monster Hunter breathing & secondary poise)
    # =========================================================================
    idle = new_clip("idle", 3.6, loop=True, description="Predatory Monster Hunter breathing stance with secondary tail, jaw, and wing/organ motion.")
    if chest_b:
        add_wave(idle, chest_b, [(0, 2.2, 0.0, 0.0)])
    apply_neck_wave(idle, 3.2, axis=0, phase0=65.0)
    if head_b:
        add_wave(idle, head_b, [(0, -2.0, 65.0, 0.0), (1, 3.5, 140.0, 0.0)])
    if jaw_b:
        add_wave(idle, jaw_b, [(0, 3.2, 20.0, 3.5)])
    apply_tail_wave(idle, 4.2 if is_leviathan else 3.2, axis=1, phase0=30.0)
    for wr in wing_roots:
        s = side_sign(wr)
        add_wave(idle, wr, [(2, s * 2.8, 90.0, 0.0), (1, -s * 1.8, 45.0, 0.0)])
    for idx, ch_b in enumerate(chains_l + chains_r):
        add_wave(idle, ch_b, [(0, 3.5 + idx * 1.2, 120.0 - idx * 35.0, 0.0), (2, 2.5, 45.0 - idx * 30.0, 0.0)])
    if mid == "abyssal_lagiacrus":
        add_wave(idle, "root", [(1, 2.2, 0.0, 4.0)], channel="position")

    # =========================================================================
    # 2. WALK (Weighted Monster Hunter stalking stride with counter-flexion)
    # =========================================================================
    walk = new_clip("walk", 1.6, loop=True, description="Weighted Monster Hunter stalking gait with full hip/knee/ankle articulation and spine roll.")
    add_wave(walk, "root", [(1, 1.1, 90.0, 0.0)], cycles=2, channel="position")
    if chest_b:
        add_wave(walk, chest_b, [(2, 4.5 if is_fanged else 3.0, 0.0, 0.0), (1, 5.5 if is_leviathan else 2.5, 90.0, 0.0)])
    if pelvis_b:
        add_wave(walk, pelvis_b, [(1, -6.0 if is_leviathan else -3.0, 90.0, 0.0), (2, -2.5, 0.0, 0.0)])
    apply_neck_wave(walk, 5.0, axis=0, cycles=2, phase0=90.0)
    if head_b:
        add_wave(walk, head_b, [(0, -4.0, 90.0, 0.0)], cycles=2)
    apply_tail_wave(walk, 8.5 if is_leviathan else 5.5, axis=1, cycles=1, phase0=40.0)

    for q, ph in (("l", 0.0), ("r", 180.0)):
        if has_hindlegs:
            h_amp = 24.0 if is_biped_wyvern else (18.0 if is_leviathan else 22.0)
            add_wave(walk, f"hind_upper_{q}", [(0, h_amp, ph, 0.0)])
            add_wave(walk, f"hind_lower_{q}", [(0, 16.0, ph + 85.0, 4.0)])
            add_wave(walk, f"hind_paw_{q}", [(0, -14.0, ph + 35.0, 0.0)])
        if has_forelegs:
            f_amp = 24.0 if is_fanged else (18.0 if is_leviathan else 21.0)
            f_ph = ph + 180.0  # Diagonal quadruped gait!
            add_wave(walk, f"fore_upper_{q}", [(0, f_amp, f_ph, 0.0)])
            add_wave(walk, f"fore_lower_{q}", [(0, -16.0, f_ph + 85.0, -4.0)])
            add_wave(walk, f"fore_paw_{q}", [(0, 13.0, f_ph + 40.0, 0.0)])
        if is_biped_wyvern and f"wing_{q}" in bnames:
            s = 1.0 if q == "l" else -1.0
            add_wave(walk, f"wing_{q}", [(1, s * 5.0, ph, 0.0), (2, s * 3.5, ph + 90.0, 0.0)])

    # =========================================================================
    # 3. RUN / BOUNDING CHARGE (Vigorous high-speed Monster Hunter pursuit!)
    # =========================================================================
    run = new_clip("run", 0.95, loop=True, description="High-energy Monster Hunter bounding charge / predatory sprint.")
    add_wave(run, "root", [(1, 3.2, 0.0, 1.2), (2, 1.8, 90.0, 0.0)], channel="position")
    if chest_b:
        add_wave(run, chest_b, [(0, 8.5, 0.0, 6.0), (2, 5.0, 90.0, 0.0)])
    if pelvis_b:
        add_wave(run, pelvis_b, [(0, -6.5, 180.0, 0.0)])
    apply_neck_wave(run, 9.5, axis=0, phase0=180.0, bias=8.0)
    if head_b:
        add_wave(run, head_b, [(0, -7.5, 180.0, -6.0)])
    if jaw_b:
        add_wave(run, jaw_b, [(0, 5.0, 0.0, 8.0)])
    apply_tail_wave(run, 11.0 if is_leviathan else 7.5, axis=1, phase0=0.0, pitch_bias=-4.0)

    if has_forelegs and has_hindlegs and not is_leviathan:
        # Bounding gallop phase (left/right pairs slightly offset by 35 deg!)
        for q, ph in (("l", 0.0), ("r", 35.0)):
            add_wave(run, f"fore_upper_{q}", [(0, 34.0, ph, 0.0)])
            add_wave(run, f"fore_lower_{q}", [(0, -24.0, ph + 75.0, -6.0)])
            add_wave(run, f"fore_paw_{q}", [(0, 18.0, ph + 35.0, 0.0)])
            add_wave(run, f"hind_upper_{q}", [(0, 32.0, ph + 175.0, 0.0)])
            add_wave(run, f"hind_lower_{q}", [(0, 24.0, ph + 250.0, 6.0)])
            add_wave(run, f"hind_paw_{q}", [(0, -18.0, ph + 210.0, 0.0)])
    else:
        for q, ph in (("l", 0.0), ("r", 180.0)):
            if has_hindlegs:
                add_wave(run, f"hind_upper_{q}", [(0, 36.0, ph, 0.0)])
                add_wave(run, f"hind_lower_{q}", [(0, 26.0, ph + 80.0, 6.0)])
                add_wave(run, f"hind_paw_{q}", [(0, -20.0, ph + 35.0, 0.0)])
            if has_forelegs:
                add_wave(run, f"fore_upper_{q}", [(0, 28.0, ph + 180.0, 0.0)])
                add_wave(run, f"fore_lower_{q}", [(0, -20.0, ph + 260.0, 0.0)])
            if f"wing_{q}" in bnames:
                s = 1.0 if q == "l" else -1.0
                add_wave(run, f"wing_{q}", [(2, s * 8.0, 0.0, s * 10.0), (1, -s * 12.0, 0.0, -s * 14.0)])

    # =========================================================================
    # 4. ALERT (Threat detection & fierce left-right sensory scan)
    # =========================================================================
    alert = new_clip("alert", 3.2, loop=False, description="Raises head sharply, scans left and right for hunters, flares crests/wings, and settles.")
    ts_al = [0.0, 0.18, 0.38, 0.62, 0.82, 1.0]
    if chest_b:
        add_curve(alert, chest_b, ts_al, [[0, 0, 0], [-10, 8, 0], [-12, 16, -4], [-12, -18, 4], [-6, 4, 0], [0, 0, 0]])
    apply_neck_curve(alert, ts_al, [0, -18, -20, -20, -10, 0], [0, 14, 30, -32, 8, 0])
    if head_b:
        add_curve(alert, head_b, ts_al, [[0, 0, 0], [8, 12, -6], [6, 24, -10], [6, -26, 10], [4, 6, 0], [0, 0, 0]])
    if jaw_b:
        add_curve(alert, jaw_b, ts_al, [[0, 0, 0], [10, 0, 0], [6, 0, 0], [14, 0, 0], [4, 0, 0], [0, 0, 0]])
    for fb in feelers:
        add_curve(alert, fb, ts_al, [[0, 0, 0], [-35, 0, 0], [-52, 0, 0], [-52, 0, 0], [-20, 0, 0], [0, 0, 0]])
    for hb in hoods:
        s = side_sign(hb)
        add_curve(alert, hb, ts_al, [[0, 0, 0], [0, -s * 24, 0], [0, -s * 34, 0], [0, -s * 34, 0], [0, -s * 12, 0], [0, 0, 0]])

    # =========================================================================
    # 5. HURT / FLINCH RECOIL (Sharp stagger hit reaction)
    # =========================================================================
    hurt = new_clip("hurt", 1.25, loop=False, description="Heavy impact flinch recoil with head whip, open jaw, and limb brace recovery.")
    ts_hu = [0.0, 0.14, 0.36, 0.68, 1.0]
    add_curve(hurt, "root", ts_hu, [[0, 0, 0], [-2.5, -1.8, -5.0], [-3.5, -2.5, -7.5], [-1.0, -0.5, -2.5], [0, 0, 0]], channel="position")
    if chest_b:
        add_curve(hurt, chest_b, ts_hu, [[0, 0, 0], [-14, -16, 10], [-18, -22, 14], [6, -6, 3], [0, 0, 0]])
    apply_neck_curve(hurt, ts_hu, [0, -24, -30, 10, 0], [0, -22, -28, -8, 0])
    if head_b:
        add_curve(hurt, head_b, ts_hu, [[0, 0, 0], [-16, -18, 18], [-22, -24, 22], [8, -6, 4], [0, 0, 0]])
    if jaw_b:
        add_curve(hurt, jaw_b, ts_hu, [[0, 0, 0], [32, 0, 0], [36, 0, 0], [12, 0, 0], [0, 0, 0]])
    for wr in wing_roots:
        s = side_sign(wr)
        add_curve(hurt, wr, ts_hu, [[0, 0, 0], [12, s * 18, s * 24], [16, s * 22, s * 28], [4, s * 6, s * 8], [0, 0, 0]])

    # =========================================================================
    # 6. ROAR (Ear-splitting Monster Hunter Enraged Roar!)
    # =========================================================================
    roar = new_clip("roar", 2.8, loop=False, description="Deep crouch wind-up into an explosive Monster Hunter ear-splitting roar with full organ/wing flare.")
    ts_ro = [0.0, 0.22, 0.42, 0.56, 0.70, 0.84, 1.0]
    add_curve(roar, "root", ts_ro, [[0, 0, 0], [0, -2.5, -2.0], [0, 2.5, 1.5], [0, 2.8, 2.0], [0, 2.4, 1.5], [0, -0.8, 0], [0, 0, 0]], channel="position")
    if chest_b:
        ch_pitch = -18 if has_forelegs else -14
        add_curve(roar, chest_b, ts_ro, [[0, 0, 0], [10, 0, 0], [ch_pitch + 2, 0, 0], [ch_pitch, 3, -2], [ch_pitch + 1, -3, 2], [4, 0, 0], [0, 0, 0]])
    apply_neck_curve(roar, ts_ro, [0, 12, -16, -18, -16, 4, 0], [0, 0, 0, 6, -6, 0, 0])
    if head_b:
        # Counter-pitch forward so jaws roar fiercely toward hunter without splitting neck/chest!
        add_curve(roar, head_b, ts_ro, [[0, 0, 0], [10, 0, 0], [16, 0, 0], [18, 6, -4], [16, -6, 4], [4, 0, 0], [0, 0, 0]])
    if jaw_b:
        add_curve(roar, jaw_b, ts_ro, [[0, 0, 0], [4, 0, 0], [38, 0, 0], [44, 0, 0], [42, 0, 0], [12, 0, 0], [0, 0, 0]])
    if has_forelegs:
        for q in ("l", "r"):
            s = 1.0 if q == "l" else -1.0
            add_curve(roar, f"fore_upper_{q}", ts_ro, [[0, 0, 0], [8, 0, -s * 8], [-22, -s * 12, s * 22], [-26, -s * 14, s * 26], [-24, -s * 12, s * 24], [-4, 0, 0], [0, 0, 0]])
            add_curve(roar, f"fore_lower_{q}", ts_ro, [[0, 0, 0], [-6, 0, 0], [18, 0, 0], [22, 0, 0], [20, 0, 0], [4, 0, 0], [0, 0, 0]])
    for wr in wing_roots:
        s = side_sign(wr)
        add_curve(roar, wr, ts_ro, [[0, 0, 0], [-10, -s * 16, -s * 18], [16, s * 24, s * 34], [20, s * 28, s * 38], [18, s * 26, s * 36], [-4, 0, 0], [0, 0, 0]])
    for wm in wing_mids + wing_dists:
        s = side_sign(wm)
        add_curve(roar, wm, ts_ro, [[0, 0, 0], [0, -s * 18, -s * 14], [0, s * 14, s * 16], [0, s * 18, s * 20], [0, s * 16, s * 18], [0, 0, 0], [0, 0, 0]])
    for fb in feelers:
        add_curve(roar, fb, ts_ro, [[0, 0, 0], [10, 0, 0], [-58, 0, 0], [-65, 0, 0], [-62, 0, 0], [-18, 0, 0], [0, 0, 0]])
    for hb in hoods:
        s = side_sign(hb)
        add_curve(roar, hb, ts_ro, [[0, 0, 0], [0, s * 8, 0], [0, -s * 38, 0], [0, -s * 44, 0], [0, -s * 42, 0], [0, -s * 10, 0], [0, 0, 0]])
    for cb in capacitors:
        add_curve(roar, cb, ts_ro, [[0, 0, 0], [6, 0, 0], [-28, 0, 0], [-34, 0, 0], [-32, 0, 0], [-8, 0, 0], [0, 0, 0]])
    for idx, tb in enumerate(tails):
        add_curve(roar, tb, ts_ro, [[0, 0, 0], [-8, 0, 0], [12, 10 if idx % 2 == 0 else -10, 0], [16, -12, 0], [14, 12, 0], [-4, 0, 0], [0, 0, 0]])

    # =========================================================================
    # 7. BITE / SAVAGE SNAP LUNGE
    # =========================================================================
    bite = new_clip("bite", 1.55, loop=False, description="Coils neck back with wide jaws, lunges explosively forward, and snaps shut.")
    ts_bi = [0.0, 0.30, 0.48, 0.62, 0.82, 1.0]
    add_curve(bite, "root", ts_bi, [[0, 0, 0], [0, -1.5, -5.5], [0, 1.0, 11.5], [0, 0.5, 13.0], [0, -0.5, 3.5], [0, 0, 0]], channel="position")
    if chest_b:
        add_curve(bite, chest_b, ts_bi, [[0, 0, 0], [-14, -10, 0], [18, 8, -6], [14, 4, -4], [-4, 0, 0], [0, 0, 0]])
    apply_neck_curve(bite, ts_bi, [0, -28, 26, 20, -6, 0], [0, -16, 12, 6, 0, 0])
    if head_b:
        add_curve(bite, head_b, ts_bi, [[0, 0, 0], [-18, -12, -14], [14, 10, 18], [10, 6, 8], [-4, 0, 0], [0, 0, 0]])
    if jaw_b:
        add_curve(bite, jaw_b, ts_bi, [[0, 0, 0], [42, 0, 0], [38, 0, 0], [-2, 0, 0], [6, 0, 0], [0, 0, 0]])

    # =========================================================================
    # 8. TAIL SWEEP (180-deg coiled Monster Hunter tail whip!)
    # =========================================================================
    tsweep = new_clip("tail_sweep", 2.1, loop=False, description="Winds body and multi-bone tail deep to one side, then whips violently through a wide arc.")
    ts_tw = [0.0, 0.28, 0.52, 0.72, 0.88, 1.0]
    if pelvis_b:
        add_curve(tsweep, pelvis_b, ts_tw, [[0, 0, 0], [0, 26, 0], [0, -34, 0], [0, -22, 0], [0, 6, 0], [0, 0, 0]])
    if chest_b:
        add_curve(tsweep, chest_b, ts_tw, [[0, 0, 0], [6, 18, 0], [8, -24, 0], [4, -14, 0], [0, 4, 0], [0, 0, 0]])
    apply_neck_curve(tsweep, ts_tw, [0, 6, 8, 4, 0, 0], [0, 22, -26, -14, 4, 0])
    for idx, tb in enumerate(tails):
        w_scale = 1.0 + idx * 0.22
        add_curve(
            tsweep,
            tb,
            ts_tw,
            [
                [0, 0, 0],
                [4, 22 * w_scale, 0],
                [8, -32 * w_scale, -8],
                [6, -36 * w_scale if idx >= 2 else -18 * w_scale, -4],
                [0, 10 * w_scale, 0],
                [0, 0, 0],
            ],
        )

    # =========================================================================
    # 9. CLADE SPECIALIZED LOCOMOTION (Wingbeat / Swim Surge / Fulgur Charge)
    # =========================================================================
    if wing_roots:
        wb = new_clip("wingbeat", 1.75, loop=True, description="Airborne power & recovery strokes with phase-delayed folding across all non-uni-sheet wing banks.")
        add_wave(wb, "root", [(1, 4.5, 220.0, 20.0), (0, 2.0, 40.0, 0.0)], channel="position")
        if chest_b:
            add_wave(wb, chest_b, [(0, 5.5, 210.0, -4.0)])
        apply_neck_wave(wb, 5.0, axis=0, phase0=30.0)
        apply_tail_wave(wb, 6.5, axis=0, phase0=120.0, pitch_bias=-6.0)
        for q in ("l", "r"):
            s = 1.0 if q == "l" else -1.0
            if f"wing_{q}" in bnames:
                add_wave(wb, f"wing_{q}", [(2, s * 32.0, 0.0, s * 4.0), (0, 10.0, 90.0, 4.0)])
            if f"wing_fore_{q}" in bnames:
                add_wave(wb, f"wing_fore_{q}", [(2, s * 18.0, -45.0, 0.0)])
            if f"wing_wrist_{q}" in bnames:
                add_wave(wb, f"wing_wrist_{q}", [(2, s * 16.0, -80.0, 0.0), (1, -s * 10.0, -60.0, 0.0)])
            if f"wing_mid_{q}" in bnames:
                add_wave(wb, f"wing_mid_{q}", [(2, s * 14.0, -110.0, 0.0)])
            if f"wing_distal_{q}" in bnames:
                add_wave(wb, f"wing_distal_{q}", [(2, s * 14.0, -140.0, 0.0)])
            if has_hindlegs:
                add_wave(wb, f"hind_upper_{q}", [(0, 5.0, 90.0, 24.0)])
                add_wave(wb, f"hind_lower_{q}", [(0, 6.0, 110.0, 28.0)])
    elif is_leviathan:
        sw = new_clip("swim_surge", 1.8, loop=True, description="Full-body pelagic/serpentine S-wave propulsion with swept limbs and paddle-tail thrust.")
        add_wave(sw, "root", [(0, 3.5, 0.0, 0.0), (1, 2.5, 90.0, 8.0)], channel="position")
        if chest_b:
            add_wave(sw, chest_b, [(1, 16.0, 0.0, 0.0), (2, 6.0, 90.0, 0.0)])
        if pelvis_b:
            add_wave(sw, pelvis_b, [(1, 18.0, -70.0, 0.0)])
        apply_neck_wave(sw, 14.0, axis=1, phase0=65.0)
        apply_tail_wave(sw, 16.0, axis=1, phase0=-130.0)
        for q in ("l", "r"):
            s = 1.0 if q == "l" else -1.0
            if has_forelegs:
                add_wave(sw, f"fore_upper_{q}", [(0, 8.0, 0.0, 38.0), (2, s * 10.0, 90.0, s * 16.0)])
            if has_hindlegs:
                add_wave(sw, f"hind_upper_{q}", [(0, 8.0, 180.0, 36.0), (2, s * 10.0, 270.0, s * 14.0)])
    else:
        chg = new_clip("thunder_charge", 3.0, loop=False, description="Plants massive forelimbs wide, arches chest skyward, and charges dorsal fulgur/dracophage organs.")
        ts_ch = [0.0, 0.25, 0.48, 0.72, 0.88, 1.0]
        add_curve(chg, "root", ts_ch, [[0, 0, 0], [0, -3.0, -1.5], [0, 1.8, 0.5], [0, 2.2, 0.8], [0, -1.0, 0], [0, 0, 0]], channel="position")
        if chest_b:
            add_curve(chg, chest_b, ts_ch, [[0, 0, 0], [16, 0, 0], [-26, 0, 0], [-30, 0, 0], [6, 0, 0], [0, 0, 0]])
        apply_neck_curve(chg, ts_ch, [0, 16, -34, -38, 8, 0])
        if head_b:
            add_curve(chg, head_b, ts_ch, [[0, 0, 0], [12, 0, 0], [-22, 0, 0], [-26, 0, 0], [6, 0, 0], [0, 0, 0]])
        if jaw_b:
            add_curve(chg, jaw_b, ts_ch, [[0, 0, 0], [6, 0, 0], [36, 0, 0], [42, 0, 0], [8, 0, 0], [0, 0, 0]])
        for cb in capacitors:
            add_curve(chg, cb, ts_ch, [[0, 0, 0], [8, 0, 0], [-36, 0, 0], [-42, 0, 0], [-10, 0, 0], [0, 0, 0]])

    # =========================================================================
    # 10, 11, 12. SPECIES-SPECIFIC WILD MONSTER HUNTER SIGNATURE ATTACKS!
    # =========================================================================
    if "zinogre" in mid:
        # Clip 10: Heavy alternating double forepaw slam!
        slam = new_clip("forepaw_double_slam", 2.2, loop=False, description="Rears and slams left giant shock-absorber forelimb, then rolls into a crushing right forepaw slam.")
        ts_s = [0.0, 0.22, 0.36, 0.58, 0.74, 0.90, 1.0]
        add_curve(slam, "root", ts_s, [[0, 0, 0], [2, 5.5, -2], [-1, -2.5, 5], [-2, 6.5, 2], [1, -3.0, 9], [0, -0.5, 2], [0, 0, 0]], channel="position")
        if chest_b:
            add_curve(slam, chest_b, ts_s, [[0, 0, 0], [-24, 16, -14], [18, -12, 12], [-26, -18, 16], [20, 14, -14], [4, 0, 0], [0, 0, 0]])
        add_curve(slam, "fore_upper_l", ts_s, [[0, 0, 0], [-62, 12, 22], [22, -6, -8], [14, 0, 0], [18, 0, 0], [4, 0, 0], [0, 0, 0]])
        add_curve(slam, "fore_lower_l", ts_s, [[0, 0, 0], [-28, 0, 0], [14, 0, 0], [8, 0, 0], [10, 0, 0], [0, 0, 0], [0, 0, 0]])
        add_curve(slam, "fore_upper_r", ts_s, [[0, 0, 0], [18, 0, 0], [12, 0, 0], [-64, -12, -22], [24, 6, 8], [4, 0, 0], [0, 0, 0]])
        add_curve(slam, "fore_lower_r", ts_s, [[0, 0, 0], [10, 0, 0], [6, 0, 0], [-28, 0, 0], [14, 0, 0], [0, 0, 0], [0, 0, 0]])
        apply_neck_curve(slam, ts_s, [0, -18, 16, -20, 18, 0, 0])
        if jaw_b:
            add_curve(slam, jaw_b, ts_s, [[0, 0, 0], [24, 0, 0], [34, 0, 0], [26, 0, 0], [38, 0, 0], [6, 0, 0], [0, 0, 0]])

        # Clip 11: Wild acrobatic corkscrew backflip tail-mace slam!
        flip = new_clip("fulgur_backflip_slam", 2.1, loop=False, description="Explosive airborne backflip corkscrew driving the giant spiked tail mace overhead into the ground!")
        ts_fl = [0.0, 0.20, 0.45, 0.65, 0.84, 1.0]
        add_curve(flip, "root", ts_fl, [[0, 0, 0], [0, -3.5, -2], [0, 22.0, 4.0], [0, 12.0, 10.0], [0, -2.5, 6.0], [0, 0, 0]], channel="position")
        if pelvis_b:
            add_curve(flip, pelvis_b, ts_fl, [[0, 0, 0], [16, 0, 0], [-52, 24, 28], [28, -16, -12], [8, 0, 0], [0, 0, 0]])
        if chest_b:
            add_curve(flip, chest_b, ts_fl, [[0, 0, 0], [18, 0, 0], [-28, 18, 16], [22, -12, -10], [6, 0, 0], [0, 0, 0]])
        for idx, tb in enumerate(tails):
            add_curve(flip, tb, ts_fl, [[0, 0, 0], [-14, 12, 0], [34 + idx * 5, -18, 0], [28 + idx * 4, 14, 0], [-10, 0, 0], [0, 0, 0]])
        for q in ("l", "r"):
            s = 1.0 if q == "l" else -1.0
            add_curve(flip, f"fore_upper_{q}", ts_fl, [[0, 0, 0], [22, 0, 0], [-48, 0, s * 18], [-22, 0, s * 12], [12, 0, 0], [0, 0, 0]])
            add_curve(flip, f"hind_upper_{q}", ts_fl, [[0, 0, 0], [-24, 0, 0], [38, 0, s * 14], [24, 0, s * 8], [-10, 0, 0], [0, 0, 0]])

        # Clip 12: Supercharged prowl idle
        c_idle = new_clip("supercharged_idle", 2.8, loop=True, description="Raised supercharged combat stance with flared dorsal capacitors and crackling head sway.")
        if chest_b:
            add_wave(c_idle, chest_b, [(0, 3.0, 0.0, -6.0)])
        for cb in capacitors:
            add_wave(c_idle, cb, [(0, 5.0, 0.0, -28.0)])
        if jaw_b:
            add_wave(c_idle, jaw_b, [(0, 4.5, 0.0, 14.0)])
        apply_tail_wave(c_idle, 7.5, axis=1, phase0=0.0, pitch_bias=8.0)

    elif mid == "rey_dau":
        # Clip 10: Iconic Railgun Horn Convergence Blast!
        rail = new_clip("railgun_horn_blast", 2.6, loop=False, description="Grounds wing-scythes wide, folds BOTH cranial railgun horns forward into a focal lightning barrel, and fires!")
        ts_r = [0.0, 0.28, 0.52, 0.66, 0.78, 0.92, 1.0]
        add_curve(rail, "root", ts_r, [[0, 0, 0], [0, -2.5, -3.0], [0, -3.0, -4.0], [0, 3.5, -14.0], [0, 1.5, -10.0], [0, -0.5, -2.0], [0, 0, 0]], channel="position")
        if chest_b:
            add_curve(rail, chest_b, ts_r, [[0, 0, 0], [16, 0, 0], [20, 0, 0], [-22, 0, 0], [-14, 0, 0], [4, 0, 0], [0, 0, 0]])
        apply_neck_curve(rail, ts_r, [0, 18, 22, -26, -16, 4, 0])
        if head_b:
            add_curve(rail, head_b, ts_r, [[0, 0, 0], [-10, 0, 0], [-14, 0, 0], [-20, 0, 0], [-10, 0, 0], [0, 0, 0], [0, 0, 0]])
        if jaw_b:
            add_curve(rail, jaw_b, ts_r, [[0, 0, 0], [18, 0, 0], [36, 0, 0], [44, 0, 0], [28, 0, 0], [6, 0, 0], [0, 0, 0]])
        for rh in rail_horns:
            s = side_sign(rh)
            # Fold wide lateral horns forward (+X pitch down/forward & inward Y yaw) to converge ahead of snout!
            add_curve(rail, rh, ts_r, [[0, 0, 0], [18, -s * 42, 0], [24, -s * 54, 0], [22, -s * 52, 0], [10, -s * 24, 0], [0, 0, 0], [0, 0, 0]])
        for wr in wing_roots:
            s = side_sign(wr)
            add_curve(rail, wr, ts_r, [[0, 0, 0], [14, s * 22, -s * 24], [18, s * 28, -s * 28], [-12, -s * 16, s * 28], [-6, -s * 8, s * 14], [0, 0, 0], [0, 0, 0]])

        # Clip 11: Fulgurite Wing-Scythe Cleave!
        cleave = new_clip("wing_scythe_cleave", 1.9, loop=False, description="Banks sharply and drags the giant curved blue fulgurite wing-blade across the ground in an arc!")
        ts_cl = [0.0, 0.30, 0.56, 0.80, 1.0]
        add_curve(cleave, "root", ts_cl, [[0, 0, 0], [-4, 8, -4], [6, 2, 12], [2, 0, 4], [0, 0, 0]], channel="position")
        if chest_b:
            add_curve(cleave, chest_b, ts_cl, [[0, 0, 0], [-16, 28, -24], [22, -36, 28], [6, -10, 8], [0, 0, 0]])
        if "wing_l" in bnames:
            add_curve(cleave, "wing_l", ts_cl, [[0, 0, 0], [-18, -28, 38], [24, 46, -36], [8, 12, -8], [0, 0, 0]])
        if "wing_r" in bnames:
            add_curve(cleave, "wing_r", ts_cl, [[0, 0, 0], [12, 18, 28], [-16, -24, -34], [-4, -8, -10], [0, 0, 0]])

        # Clip 12: Takeoff Launch
        tk = new_clip("takeoff_launch", 2.2, loop=False, description="Crouches deeply and launches skyward with a thunderous wing stroke.")
        ts_tk = [0.0, 0.28, 0.55, 0.80, 1.0]
        add_curve(tk, "root", ts_tk, [[0, 0, 0], [0, -4.5, -2], [0, 22.0, 4], [0, 16.0, 2], [0, 0, 0]], channel="position")
        for wr in wing_roots:
            s = side_sign(wr)
            add_curve(tk, wr, ts_tk, [[0, 0, 0], [-12, -s * 14, s * 36], [18, s * 16, -s * 38], [8, 0, s * 22], [0, 0, 0]])

    elif mid == "arkveld":
        # Clip 10: Overhead Dual Chainblade Crushing Slam!
        cs = new_clip("chainblade_overhead_slam", 2.3, loop=False, description="Rears tall on hindlegs, whips multi-link wrist chainblades high overhead, and crashes them forward!")
        ts_cs = [0.0, 0.34, 0.56, 0.74, 0.90, 1.0]
        add_curve(cs, "root", ts_cs, [[0, 0, 0], [0, 8.5, -4], [0, -3.0, 8], [0, -2.0, 6], [0, -0.5, 1], [0, 0, 0]], channel="position")
        if chest_b:
            add_curve(cs, chest_b, ts_cs, [[0, 0, 0], [-36, 0, 0], [26, 0, 0], [18, 0, 0], [4, 0, 0], [0, 0, 0]])
        apply_neck_curve(cs, ts_cs, [0, -22, 24, 16, 4, 0])
        if jaw_b:
            add_curve(cs, jaw_b, ts_cs, [[0, 0, 0], [32, 0, 0], [40, 0, 0], [18, 0, 0], [4, 0, 0], [0, 0, 0]])
        for wr in wing_roots:
            s = side_sign(wr)
            add_curve(cs, wr, ts_cs, [[0, 0, 0], [-42, s * 18, s * 36], [32, -s * 14, -s * 24], [22, -s * 10, -s * 16], [4, 0, 0], [0, 0, 0]])
        for idx, ch_b in enumerate(chains_l):
            add_curve(cs, ch_b, ts_cs, [[0, 0, 0], [-38 - idx * 8, -18, 12], [44 + idx * 6, 12, -10], [24, 6, -4], [6, 0, 0], [0, 0, 0]])
        for idx, ch_b in enumerate(chains_r):
            add_curve(cs, ch_b, ts_cs, [[0, 0, 0], [-38 - idx * 8, 18, -12], [44 + idx * 6, -12, 10], [24, -6, 4], [6, 0, 0], [0, 0, 0]])

        # Clip 11: Wide Cross-Body Chainblade Sweep!
        swp = new_clip("chainblade_whip_sweep", 2.0, loop=False, description="Pivots torso and whips multi-segment wrist chainblades across a huge horizontal cleaving arc!")
        ts_sw = [0.0, 0.28, 0.54, 0.78, 1.0]
        if chest_b:
            add_curve(swp, chest_b, ts_sw, [[0, 0, 0], [-12, 28, -10], [16, -34, 14], [6, -12, 4], [0, 0, 0]])
        if "wing_l" in bnames:
            add_curve(swp, "wing_l", ts_sw, [[0, 0, 0], [-18, -32, 28], [18, 48, -22], [6, 14, -6], [0, 0, 0]])
        for idx, ch_b in enumerate(chains_l):
            add_curve(swp, ch_b, ts_sw, [[0, 0, 0], [-10, -28 - idx * 6, 14], [16, 42 + idx * 8, -16], [8, 18, -6], [0, 0, 0]])
        for idx, ch_b in enumerate(chains_r):
            add_curve(swp, ch_b, ts_sw, [[0, 0, 0], [-10, -20, -10], [14, -36 - idx * 6, 14], [6, -12, 4], [0, 0, 0]])

        # Clip 12: Wraith Aerial Pounce
        pnc = new_clip("wraith_pounce_crush", 2.1, loop=False, description="Bounds into the air trailing white fur and chainblades, slamming down onto prey.")
        ts_p = [0.0, 0.24, 0.52, 0.78, 1.0]
        add_curve(pnc, "root", ts_p, [[0, 0, 0], [0, -3.5, -3], [0, 18.0, 8], [0, -2.5, 14], [0, 0, 0]], channel="position")
        if chest_b:
            add_curve(pnc, chest_b, ts_p, [[0, 0, 0], [16, 0, 0], [-22, 0, 0], [20, 0, 0], [0, 0, 0]])

    elif "gore_magala" in mid:
        # Clip 10: Frenzy Eclipse Feelers & Mantle Transformation!
        frz = new_clip("frenzy_eclipse_display", 2.8, loop=False, description="Throws tattered mantle wingarms skyward like a dark archangel while unfurling glowing Frenzy feeler horns!")
        ts_f = [0.0, 0.25, 0.50, 0.74, 0.90, 1.0]
        add_curve(frz, "root", ts_f, [[0, 0, 0], [0, -3.0, -2], [0, 9.0, -3], [0, 8.5, -2], [0, -1.0, 0], [0, 0, 0]], channel="position")
        if chest_b:
            add_curve(frz, chest_b, ts_f, [[0, 0, 0], [16, 0, 0], [-38, 0, 0], [-36, 0, 0], [6, 0, 0], [0, 0, 0]])
        apply_neck_curve(frz, ts_f, [0, 18, -26, -24, 6, 0])
        if jaw_b:
            add_curve(frz, jaw_b, ts_f, [[0, 0, 0], [6, 0, 0], [40, 0, 0], [42, 0, 0], [8, 0, 0], [0, 0, 0]])
        for fb in feelers:
            add_curve(frz, fb, ts_f, [[0, 0, 0], [8, 0, 0], [-68, 0, 0], [-72, 0, 0], [-20, 0, 0], [0, 0, 0]])
        for wr in wing_roots:
            s = side_sign(wr)
            add_curve(frz, wr, ts_f, [[0, 0, 0], [10, -s * 18, -s * 16], [-28, s * 26, s * 44], [-26, s * 24, s * 42], [4, 0, 0], [0, 0, 0]])

        # Clip 11: Colossal Wingarm Ground Slam (asymmetric combo on Chaotic Gore!)
        wsl = new_clip("wingarm_frenzy_slam", 2.2, loop=False, description="Rears up and smashes giant clawed wingarm fists violently into the earth!")
        ts_w = [0.0, 0.32, 0.54, 0.76, 1.0]
        add_curve(wsl, "root", ts_w, [[0, 0, 0], [0, 9.5, -3], [0, -3.5, 9], [0, -2.0, 6], [0, 0, 0]], channel="position")
        if chest_b:
            add_curve(wsl, chest_b, ts_w, [[0, 0, 0], [-38, 12 if "chaotic" in mid else 0, 0], [28, -14 if "chaotic" in mid else 0, 0], [16, 0, 0], [0, 0, 0]])
        for wr in wing_roots:
            s = side_sign(wr)
            asym = 1.25 if ("chaotic" in mid and s < 0) else 1.0
            add_curve(wsl, wr, ts_w, [[0, 0, 0], [-44 * asym, s * 22, s * 38], [34 * asym, -s * 24, -s * 28], [18, -s * 10, -s * 14], [0, 0, 0]])

        # Clip 12: Blind Sensory Prowl Scan
        scn = new_clip("frenzy_sensory_scan", 2.9, loop=False, description="Raises feelers and sweeps eyeless visor skull low across the ground tasting Frenzy spores.")
        ts_sc = [0.0, 0.25, 0.52, 0.78, 1.0]
        apply_neck_curve(scn, ts_sc, [0, 22, 24, 22, 0], [0, 28, -30, 14, 0])
        for fb in feelers:
            add_curve(scn, fb, ts_sc, [[0, 0, 0], [-56, 0, 0], [-60, 0, 0], [-54, 0, 0], [0, 0, 0]])

    elif mid == "velkhana":
        # Clip 10: Signature Overhead Scorpion Tail-Lance Triple Flurry!
        lance = new_clip("scorpion_tail_lance_flurry", 2.4, loop=False, description="Arches whip tail 80-deg scorpion-style right over its back and stabs forward with its ice-rapier lance!")
        ts_ln = [0.0, 0.24, 0.40, 0.54, 0.68, 0.84, 1.0]
        if chest_b:
            add_curve(lance, chest_b, ts_ln, [[0, 0, 0], [14, 0, 0], [18, -6, 0], [16, 6, 0], [20, 0, 0], [6, 0, 0], [0, 0, 0]])
        apply_neck_curve(lance, ts_ln, [0, -14, -18, -16, -18, -6, 0])
        for idx, tb in enumerate(tails):
            # +X pitches -Z tail UP and OVERHEAD toward +Z!
            arch = 36.0 if idx < 2 else 28.0
            stab = 44.0 if idx < 2 else 34.0
            add_curve(lance, tb, ts_ln, [[0, 0, 0], [arch, 0, 0], [stab, -6, 0], [arch - 6, 6, 0], [stab + 4, 0, 0], [12, 0, 0], [0, 0, 0]])
        for wr in wing_roots:
            s = side_sign(wr)
            add_curve(lance, wr, ts_ln, [[0, 0, 0], [8, s * 18, s * 26], [10, s * 22, s * 30], [8, s * 18, s * 26], [12, s * 24, s * 32], [0, 0, 0], [0, 0, 0]])

        # Clip 11: Sweeping Subzero Frost Breath
        fb_clip = new_clip("subzero_frost_breath", 2.6, loop=False, description="Rears gracefully and sweeps an icy absolute-zero breath beam left to right.")
        ts_fb = [0.0, 0.25, 0.52, 0.78, 1.0]
        if chest_b:
            add_curve(fb_clip, chest_b, ts_fb, [[0, 0, 0], [-18, 22, 0], [-14, -26, 0], [-10, 8, 0], [0, 0, 0]])
        apply_neck_curve(fb_clip, ts_fb, [0, -16, 12, 8, 0], [0, 28, -32, 10, 0])
        if jaw_b:
            add_curve(fb_clip, jaw_b, ts_fb, [[0, 0, 0], [34, 0, 0], [38, 0, 0], [32, 0, 0], [0, 0, 0]])

        # Clip 12: Absolute Zero Aerial Spire Nova
        nv = new_clip("absolute_zero_nova", 2.7, loop=False, description="Ascends into midair, spreads crystalline frost wings wide, and unleashes radial rime spires.")
        ts_nv = [0.0, 0.34, 0.62, 0.84, 1.0]
        add_curve(nv, "root", ts_nv, [[0, 0, 0], [0, 24.0, 0], [0, 28.0, 0], [0, 8.0, 0], [0, 0, 0]], channel="position")
        for wr in wing_roots:
            s = side_sign(wr)
            add_curve(nv, wr, ts_nv, [[0, 0, 0], [-14, s * 18, s * 42], [18, s * 26, -s * 24], [6, 0, s * 14], [0, 0, 0]])

    elif mid == "crimson_glow_valstrax":
        # Clip 10: Supersonic Around-The-World Comet Ambush!
        cmt = new_clip("supersonic_comet_ambush", 2.5, loop=False, description="Rotates 3-nozzle trident jet wings backward into supersonic delta-jet formation and blasts forward!")
        ts_cm = [0.0, 0.28, 0.50, 0.74, 0.90, 1.0]
        add_curve(cmt, "root", ts_cm, [[0, 0, 0], [0, 18.0, -8], [0, 10.0, 24], [0, 4.0, 16], [0, -1.5, 4], [0, 0, 0]], channel="position")
        if chest_b:
            add_curve(cmt, chest_b, ts_cm, [[0, 0, 0], [-18, 0, 0], [26, 0, 0], [22, 0, 0], [4, 0, 0], [0, 0, 0]])
        apply_neck_curve(cmt, ts_cm, [0, -12, 22, 18, 4, 0])
        for wr in wing_roots:
            s = side_sign(wr)
            # Sweep wings sharply aft into delta jet nozzles!
            add_curve(cmt, wr, ts_cm, [[0, 0, 0], [-12, s * 24, s * 32], [16, -s * 52, -s * 12], [14, -s * 48, -s * 10], [4, -s * 12, 0], [0, 0, 0]])

        # Clip 11: Trident Wing-Lance Thrust!
        lnc = new_clip("trident_wing_lance_thrust", 1.95, loop=False, description="Flips right trident wing manifold forward over the shoulder and stabs it forward like a giant spear!")
        ts_lt = [0.0, 0.30, 0.54, 0.78, 1.0]
        if chest_b:
            add_curve(lnc, chest_b, ts_lt, [[0, 0, 0], [-10, -24, 0], [14, 32, -8], [6, 10, 0], [0, 0, 0]])
        if "wing_r" in bnames:
            add_curve(lnc, "wing_r", ts_lt, [[0, 0, 0], [-20, 24, -28], [18, -62, 16], [8, -22, 6], [0, 0, 0]])

        # Clip 12: Dragonheart Crimson Exhaust Barrage
        exh = new_clip("crimson_exhaust_barrage", 2.3, loop=False, description="Aims both 3-nozzle jet manifolds forward to fire twin crimson dragon-energy beams.")
        ts_ex = [0.0, 0.32, 0.64, 0.86, 1.0]
        add_curve(exh, "root", ts_ex, [[0, 0, 0], [0, -2.0, 2], [0, 1.5, -8], [0, 0, -3], [0, 0, 0]], channel="position")
        for wr in wing_roots:
            s = side_sign(wr)
            add_curve(exh, wr, ts_ex, [[0, 0, 0], [12, s * 54, -s * 14], [14, s * 58, -s * 16], [6, s * 20, -s * 6], [0, 0, 0]])

    elif mid == "ruiner_nergigante":
        # Clip 10: Heaven-Ruin Extinction Divebomb!
        dvb = new_clip("heaven_ruin_divebomb", 2.4, loop=False, description="Soars skyward flaring spiked wings, roars, then rockets headfirst down in an iron-spike shattering slam!")
        ts_dv = [0.0, 0.32, 0.50, 0.68, 0.86, 1.0]
        add_curve(dvb, "root", ts_dv, [[0, 0, 0], [0, 26.0, -6], [0, 28.0, -4], [0, -3.0, 16], [0, -1.5, 8], [0, 0, 0]], channel="position")
        if chest_b:
            add_curve(dvb, chest_b, ts_dv, [[0, 0, 0], [-26, 0, 0], [-18, 0, 0], [32, 0, 0], [10, 0, 0], [0, 0, 0]])
        for wr in wing_roots:
            s = side_sign(wr)
            add_curve(dvb, wr, ts_dv, [[0, 0, 0], [-16, s * 24, s * 44], [-12, s * 28, s * 48], [22, -s * 34, -s * 24], [8, -s * 10, -s * 8], [0, 0, 0]])
        for q in ("l", "r"):
            add_curve(dvb, f"fore_upper_{q}", ts_dv, [[0, 0, 0], [-42, 0, 0], [-48, 0, 0], [28, 0, 0], [10, 0, 0], [0, 0, 0]])

        # Clip 11: Brutal Iron-Spike Forearm Smash
        fsm = new_clip("ironspike_forearm_smash", 2.0, loop=False, description="Rears on hindlegs raising massive iron-spiked right gauntlet and drives it through the floor!")
        ts_fs = [0.0, 0.34, 0.56, 0.80, 1.0]
        if chest_b:
            add_curve(fsm, chest_b, ts_fs, [[0, 0, 0], [-34, -22, 16], [24, 24, -18], [8, 8, -4], [0, 0, 0]])
        add_curve(fsm, "fore_upper_r", ts_fs, [[0, 0, 0], [-68, -18, -28], [26, 10, 12], [10, 4, 4], [0, 0, 0]])

        # Clip 12: Spike-Shatter Shoulder Quake
        shk = new_clip("spike_shatter_shake", 2.1, loop=False, description="Plants heavy forelimbs and violently snaps spiked shoulders side-to-side launching iron thorns.")
        ts_sk = [0.0, 0.22, 0.42, 0.62, 0.82, 1.0]
        if chest_b:
            add_curve(shk, chest_b, ts_sk, [[0, 0, 0], [10, 18, 22], [10, -20, -24], [10, 20, 24], [6, -10, -12], [0, 0, 0]])

    elif "fatalis" in mid:
        # Clip 10: Castle Schrade Upright Hellfire Cone / Crimson Lightning!
        inf = new_clip("schrade_demise_inferno", 3.0, loop=False, description="Rears upright on hindlegs, spreads colossal wings, and arches 3-bone S-neck downward spewing Hellfire!")
        ts_in = [0.0, 0.28, 0.52, 0.76, 0.90, 1.0]
        add_curve(inf, "root", ts_in, [[0, 0, 0], [0, 12.0, -4], [0, 14.0, -3], [0, 13.0, -3], [0, 2.0, -1], [0, 0, 0]], channel="position")
        if chest_b:
            add_curve(inf, chest_b, ts_in, [[0, 0, 0], [-42, 0, 0], [-38, 0, 0], [-38, 0, 0], [-10, 0, 0], [0, 0, 0]])
        apply_neck_curve(inf, ts_in, [0, 28, 42, 40, 12, 0], [0, 0, 10, -10, 0, 0])
        if head_b:
            add_curve(inf, head_b, ts_in, [[0, 0, 0], [14, 0, 0], [22, 8, 0], [22, -8, 0], [6, 0, 0], [0, 0, 0]])
        if jaw_b:
            add_curve(inf, jaw_b, ts_in, [[0, 0, 0], [28, 0, 0], [44, 0, 0], [44, 0, 0], [12, 0, 0], [0, 0, 0]])
        for wr in wing_roots:
            s = side_sign(wr)
            add_curve(inf, wr, ts_in, [[0, 0, 0], [-10, s * 24, s * 36], [-8, s * 28, s * 42], [-8, s * 28, s * 42], [0, 0, 0], [0, 0, 0]])

        # Clip 11: Wide Serpentine Flame Sweep
        fsw = new_clip("sweeping_hellfire_arc", 2.6, loop=False, description="Drops onto all fours and sweeps the long S-curved neck across a 140-degree inferno arc.")
        ts_fw = [0.0, 0.26, 0.54, 0.80, 1.0]
        apply_neck_curve(fsw, ts_fw, [0, 18, 22, 18, 0], [0, 42, -44, 16, 0])
        if jaw_b:
            add_curve(fsw, jaw_b, ts_fw, [[0, 0, 0], [38, 0, 0], [42, 0, 0], [36, 0, 0], [0, 0, 0]])

        # Clip 12: Ancestral Sky Ascent (Especially iconic for White Fatalis!)
        asc = new_clip("eclipse_sky_ascent", 2.8, loop=False, description="Rises majestically into the storm clouds with antlers/horns raised calling ancestral judgement.")
        ts_as = [0.0, 0.35, 0.68, 0.88, 1.0]
        add_curve(asc, "root", ts_as, [[0, 0, 0], [0, 26.0, 0], [0, 30.0, 0], [0, 10.0, 0], [0, 0, 0]], channel="position")
        apply_neck_curve(asc, ts_as, [0, -24, -28, -10, 0])

    elif mid == "zoh_shia":
        # Clip 10: Wylk Wingarm Cataclysm Shatter!
        cat = new_clip("wylk_wingarm_cataclysm", 2.5, loop=False, description="Raises colossal triple-clawed Wylk shroud wingarms skyward and slams them down detonating Wylk crystals!")
        ts_ct = [0.0, 0.34, 0.56, 0.78, 1.0]
        add_curve(cat, "root", ts_ct, [[0, 0, 0], [0, 10.0, -3], [0, -3.5, 8], [0, -2.0, 5], [0, 0, 0]], channel="position")
        if chest_b:
            add_curve(cat, chest_b, ts_ct, [[0, 0, 0], [-36, 0, 0], [28, 0, 0], [16, 0, 0], [0, 0, 0]])
        for wr in wing_roots:
            s = side_sign(wr)
            add_curve(cat, wr, ts_ct, [[0, 0, 0], [-42, s * 22, s * 40], [34, -s * 24, -s * 28], [16, -s * 10, -s * 12], [0, 0, 0]])

        # Clip 11: False Seraph Core Detonation
        det = new_clip("seraph_core_detonation", 2.7, loop=False, description="Spreads Wylk shroud fans wide exposing the blazing crimson chest core in a cataclysmic nova.")
        ts_dt = [0.0, 0.30, 0.62, 0.86, 1.0]
        if chest_b:
            add_curve(det, chest_b, ts_dt, [[0, 0, 0], [-28, 0, 0], [-32, 0, 0], [-10, 0, 0], [0, 0, 0]])
        for wr in wing_roots:
            s = side_sign(wr)
            add_curve(det, wr, ts_dt, [[0, 0, 0], [-16, s * 32, s * 44], [-18, s * 36, s * 46], [-6, s * 10, s * 14], [0, 0, 0]])

        # Clip 12: Black Dragon Mutation Thrash
        mut = new_clip("black_dragon_mutation_thrash", 2.2, loop=False, description="Violent asymmetric convulsion as dark draconic corruption surges through the Wylk construct.")
        ts_mt = [0.0, 0.22, 0.46, 0.72, 1.0]
        apply_neck_curve(mut, ts_mt, [0, -26, 22, -20, 0], [0, 28, -32, 24, 0])

    elif mid == "mizutsune":
        # Clip 10: Bewitching Bubble-Drift Pirouette Spin!
        spn = new_clip("bubble_drift_pirouette", 2.1, loop=False, description="Graceful sliding 360-degree dancer spin whipping the plumed kitsune brush tail through soap bubbles!")
        ts_sp = [0.0, 0.25, 0.50, 0.75, 1.0]
        add_curve(spn, "root", ts_sp, [[0, 0, 0], [6, 2.0, 4], [0, 3.0, 10], [-6, 2.0, 4], [0, 0, 0]], channel="position")
        add_curve(spn, "root", ts_sp, [[0, 0, 0], [0, 95, 0], [0, 190, 0], [0, 285, 0], [0, 360, 0]])
        apply_tail_wave(spn, 22.0, axis=1, cycles=1, phase0=0.0, pitch_bias=14.0)

        # Clip 11: Sweeping High-Pressure Water Jet
        jet = new_clip("sweeping_water_beam", 2.5, loop=False, description="Rears tall onto hindquarters and sweeps a high-pressure water beam gracefully across the arena.")
        ts_jt = [0.0, 0.26, 0.54, 0.80, 1.0]
        if chest_b:
            add_curve(jet, chest_b, ts_jt, [[0, 0, 0], [-28, 24, 0], [-26, -28, 0], [-12, 8, 0], [0, 0, 0]])
        apply_neck_curve(jet, ts_jt, [0, 14, 16, 8, 0], [0, 28, -32, 10, 0])
        if jaw_b:
            add_curve(jet, jaw_b, ts_jt, [[0, 0, 0], [32, 0, 0], [34, 0, 0], [28, 0, 0], [0, 0, 0]])

        # Clip 12: Kitsune Tail-Brush Slam
        tsl = new_clip("kitsune_brush_slam", 1.95, loop=False, description="Leaps and flips forward slamming the voluminous violet brush tail down.")
        ts_tl = [0.0, 0.28, 0.56, 0.82, 1.0]
        add_curve(tsl, "root", ts_tl, [[0, 0, 0], [0, 14.0, 2], [0, 6.0, 10], [0, -1.5, 6], [0, 0, 0]], channel="position")
        for idx, tb in enumerate(tails):
            add_curve(tsl, tb, ts_tl, [[0, 0, 0], [-24, 0, 0], [36 + idx * 5, 0, 0], [18, 0, 0], [0, 0, 0]])

    elif mid == "jin_dahaad":
        # Clip 10: Iceshard Supernova Heat-Sink Discharge!
        nov = new_clip("iceshard_nova_discharge", 2.9, loop=False, description="Coils colossal glacial body, flares every copper radiator manifold wide open, and erupts in an ice nova!")
        ts_jn = [0.0, 0.30, 0.56, 0.80, 1.0]
        if chest_b:
            add_curve(nov, chest_b, ts_jn, [[0, 0, 0], [18, 0, 0], [-30, 0, 0], [-26, 0, 0], [0, 0, 0]])
        apply_neck_curve(nov, ts_jn, [0, 22, -34, -28, 0])
        if jaw_b:
            add_curve(nov, jaw_b, ts_jn, [[0, 0, 0], [8, 0, 0], [42, 0, 0], [38, 0, 0], [0, 0, 0]])
        for cb in capacitors:
            add_curve(nov, cb, ts_jn, [[0, 0, 0], [10, 0, 0], [-42, 0, 0], [-38, 0, 0], [0, 0, 0]])

        # Clip 11: Glacial Frost Breath Sweep
        gfb = new_clip("glacial_frost_sweep", 2.5, loop=False, description="Sweeps heavy bronze-crowned head low freezing the ground in a wide arc.")
        ts_gf = [0.0, 0.28, 0.56, 0.82, 1.0]
        apply_neck_curve(gfb, ts_gf, [0, 20, 22, 18, 0], [0, 32, -34, 12, 0])
        if jaw_b:
            add_curve(gfb, jaw_b, ts_gf, [[0, 0, 0], [34, 0, 0], [38, 0, 0], [30, 0, 0], [0, 0, 0]])

        # Clip 12: Colossal Juggernaut Body Press
        prs = new_clip("juggernaut_body_press", 2.3, loop=False, description="Rears heavy forequarters high and crashes all four colossal limbs down shattering rime ice.")
        ts_pr = [0.0, 0.36, 0.58, 0.82, 1.0]
        add_curve(prs, "root", ts_pr, [[0, 0, 0], [0, 9.0, -2], [0, -3.5, 6], [0, -1.5, 4], [0, 0, 0]], channel="position")
        if chest_b:
            add_curve(prs, chest_b, ts_pr, [[0, 0, 0], [-34, 0, 0], [22, 0, 0], [8, 0, 0], [0, 0, 0]])

    elif "lagiacrus" in mid:
        # Clip 10: Cobra-Hood Thunder Dynamo AoE Discharge!
        dis = new_clip("leviathan_thunder_discharge", 2.7, loop=False, description="Coils serpentine torso, flares spiked cobra neck-hood wide, and unleashes a massive electric AoE surge!")
        ts_ds = [0.0, 0.28, 0.52, 0.76, 1.0]
        if chest_b:
            add_curve(dis, chest_b, ts_ds, [[0, 0, 0], [12, 14, 0], [-18, -10, 0], [-16, 10, 0], [0, 0, 0]])
        apply_neck_curve(dis, ts_ds, [0, 12, -22, -18, 0])
        if head_b:
            add_curve(dis, head_b, ts_ds, [[0, 0, 0], [10, 0, 0], [18, 0, 0], [16, 0, 0], [0, 0, 0]])
        if jaw_b:
            add_curve(dis, jaw_b, ts_ds, [[0, 0, 0], [8, 0, 0], [40, 0, 0], [38, 0, 0], [0, 0, 0]])
        for q in ("l", "r"):
            s = 1.0 if q == "l" else -1.0
            if f"fore_upper_{q}" in bnames:
                add_curve(dis, f"fore_upper_{q}", ts_ds, [[0, 0, 0], [10, 0, 0], [-18, -s * 14, s * 22], [-16, -s * 12, s * 20], [0, 0, 0]])
        for hb in hoods:
            s = side_sign(hb)
            add_curve(dis, hb, ts_ds, [[0, 0, 0], [0, s * 8, 0], [0, -s * 44, 0], [0, -s * 46, 0], [0, 0, 0]])
        for cb in capacitors:
            add_curve(dis, cb, ts_ds, [[0, 0, 0], [6, 0, 0], [-30, 0, 0], [-28, 0, 0], [0, 0, 0]])

        # Clip 11: Serpentine Hipcheck / Body Slide
        hip = new_clip("leviathan_body_check", 1.9, loop=False, description="Coils into an S-bend and slams its armored flank sideways in a classic Leviathan body-check!")
        ts_hp = [0.0, 0.30, 0.54, 0.80, 1.0]
        add_curve(hip, "root", ts_hp, [[0, 0, 0], [-6, 0, 0], [16, 1.5, 2], [8, 0, 1], [0, 0, 0]], channel="position")
        if chest_b:
            add_curve(hip, chest_b, ts_hp, [[0, 0, 0], [0, -24, 8], [0, 28, -14], [0, 10, -4], [0, 0, 0]])

        # Clip 12: Hadal Maelstrom Corkscrew Dive
        ml = new_clip("maelstrom_spiral_dive", 2.2, loop=False, description="Lunges forward in a hydrodynamic corkscrew roll driving its paddle tail.")
        ts_ml = [0.0, 0.25, 0.50, 0.75, 1.0]
        add_curve(ml, "root", ts_ml, [[0, 0, 0], [0, 8.0, 6], [0, 10.0, 14], [0, 4.0, 6], [0, 0, 0]], channel="position")
        if chest_b:
            add_curve(ml, chest_b, ts_ml, [[0, 0, 0], [-10, 0, 90], [0, 0, 180], [10, 0, 270], [0, 0, 360]])

    else:
        # Flying Wyverns (Rathalos, Azure Rathalos, Silver Rathalos, Legiana, Shrieking Legiana)
        is_leg = "legiana" in mid
        # Clip 10: Signature Aerial Talon Dive / Glacial Corkscrew!
        dive_name = "glacial_corkscrew_dive" if is_leg else "aerial_venom_talon_dive"
        dv = new_clip(dive_name, 2.3, loop=False, description="Ascends high, banks sharply, and rockets forward feet-first with splayed talons / corkscrew roll!")
        ts_d = [0.0, 0.30, 0.56, 0.78, 1.0]
        add_curve(dv, "root", ts_d, [[0, 0, 0], [0, 24.0, -8], [0, 8.0, 16], [0, 2.0, 10], [0, 0, 0]], channel="position")
        if chest_b:
            roll_pk = 360.0 if is_leg else 0.0
            add_curve(dv, chest_b, ts_d, [[0, 0, 0], [-28, 0, roll_pk * 0.25], [18, 0, roll_pk * 0.75], [8, 0, roll_pk], [0, 0, roll_pk]])
        for q in ("l", "r"):
            s = 1.0 if q == "l" else -1.0
            if has_hindlegs:
                add_curve(dv, f"hind_upper_{q}", ts_d, [[0, 0, 0], [22, 0, 0], [-48, -s * 16, -s * 14], [-24, -s * 8, -s * 6], [0, 0, 0]])
                add_curve(dv, f"hind_lower_{q}", ts_d, [[0, 0, 0], [24, 0, 0], [-26, 0, 0], [-12, 0, 0], [0, 0, 0]])
            if f"wing_{q}" in bnames:
                add_curve(dv, f"wing_{q}", ts_d, [[0, 0, 0], [-14, s * 20, s * 38], [16, -s * 18, -s * 32], [6, 0, s * 12], [0, 0, 0]])

        # Clip 11: Fireball / Frost-Wave Breath Barrage!
        br_name = "frost_gale_breath" if is_leg else "fireball_inferno_blast"
        br = new_clip(br_name, 2.2, loop=False, description="Pulls chest back inhaling deeply, then snaps forward blasting fire/frost with recoil!")
        ts_b = [0.0, 0.34, 0.52, 0.74, 1.0]
        add_curve(br, "root", ts_b, [[0, 0, 0], [0, -1.5, -4], [0, 2.0, -9], [0, 0.5, -4], [0, 0, 0]], channel="position")
        if chest_b:
            add_curve(br, chest_b, ts_b, [[0, 0, 0], [-22, 0, 0], [20, 0, 0], [10, 0, 0], [0, 0, 0]])
        apply_neck_curve(br, ts_b, [0, -24, 24, 14, 0])
        if jaw_b:
            add_curve(br, jaw_b, ts_b, [[0, 0, 0], [18, 0, 0], [44, 0, 0], [34, 0, 0], [0, 0, 0]])

        # Clip 12: Takeoff Launch
        tk = new_clip("takeoff_launch", 2.2, loop=False, description="Crouches deeply and launches skyward with a powerful downward wingbeat.")
        ts_tk = [0.0, 0.28, 0.55, 0.80, 1.0]
        add_curve(tk, "root", ts_tk, [[0, 0, 0], [0, -4.0, -2], [0, 22.0, 4], [0, 14.0, 2], [0, 0, 0]], channel="position")
        for wr in wing_roots:
            s = side_sign(wr)
            add_curve(tk, wr, ts_tk, [[0, 0, 0], [-12, -s * 14, s * 36], [18, s * 16, -s * 38], [8, 0, s * 22], [0, 0, 0]])

    return clips
