"""Re-derive the root trajectory of a decoded rollout from foot contact.

Why: the LM emits the trajectory as per-frame world-translation deltas that are
integrated at decode time with nothing tying the body to the ground.  Two failure
modes follow (measured on the NCSA val sessions):

* the pelvis can rise a metre and stay there when a burst of acceleration is read
  as the body leaving the ground (``seq_11``: +1.25 m, and +2.35 m for the
  noise-free checkpoint), a state the pretraining motion never contains;
* greedy decoding of the delta tokens regresses to the mode, so walking clips
  travel ~1.4x too short.

Both are fixed by the body itself: the *local* pose is accurate (root-relative
MPJPE 31-74 mm), so the stance foot gives the root velocity, and the stance foot
on the ground plane gives the root height.  Nothing here needs ground truth, a
new head or retraining: it consumes only the decoded ``pose`` / ``orient``, the
predicted ``ground`` object and the LM trajectory (kept as the fallback while no
foot is in contact).

Usage (imu4d env, repository root)::

    python -m evaluation.footlock_traj --rollout <run>/evaluation/full/.../rollout.pkl \
        --out-suffix _footlock

writes ``rollout_footlock.pkl`` next to each input and prints the metrics.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, filtfilt

# SMPL-X body joints used as feet: ankles and toes.
FOOT_JOINTS = (7, 8, 10, 11)
LEFT_FOOT = (7, 10)
RIGHT_FOOT = (8, 11)


def forward_kinematics(orient: np.ndarray, pose: np.ndarray,
                       model_path: str = "data/models") -> np.ndarray:
    """Body joints in the root frame (pelvis at the origin), [T, 22, 3].

    Uses the ``BodyModel`` that ``utils.metrics`` already keeps on the GPU for
    MPJPE when it is importable (the evaluator path), and falls back to a local
    ``smplx`` model otherwise (the offline path, CPU).
    """
    n_time = len(orient)
    try:
        from utils.human import load_smplx_model
        global _BODY_MODEL
        if _BODY_MODEL is None:
            _BODY_MODEL = load_smplx_model()
        device = next(_BODY_MODEL.parameters()).device
        with torch.no_grad():
            out = _BODY_MODEL(
                pose_body=torch.as_tensor(pose, dtype=torch.float32, device=device).reshape(-1, 63),
                root_orient=torch.as_tensor(orient, dtype=torch.float32, device=device).reshape(-1, 3),
                trans=torch.zeros(n_time, 3, device=device),
                betas=torch.zeros(n_time, 10, device=device),
            )
        joints = out.Jtr[:, :22].detach().float().cpu().numpy()
    except Exception:  # offline use without the training environment
        import smplx
        model = smplx.create(model_path, model_type="smplx", gender="neutral", ext="npz",
                             use_pca=False, batch_size=n_time)
        with torch.no_grad():
            out = model(global_orient=torch.tensor(orient).float(),
                        body_pose=torch.tensor(pose).float(),
                        transl=torch.zeros(n_time, 3))
        joints = out.joints.numpy()[:, :22]
    return joints - joints[:, 0:1]


_BODY_MODEL = None


def contact_weights(joints: np.ndarray, fps: float, height_scale: float = 0.05,
                    speed_scale: float = 0.20) -> tuple[np.ndarray, np.ndarray]:
    """Soft stance weight per foot joint, [T, 4], and the airborne-free confidence.

    The height term is relative to the *lowest* foot of the frame rather than to
    the ground plane: while walking one foot is planted at every instant, and the
    lowest-foot reference keeps that true without trusting the predicted ground
    height.  The ground plane is only used for the vertical channel.  Confidence
    drops when even the slowest foot moves fast, i.e. the body is airborne.
    """
    feet = joints[:, FOOT_JOINTS]                                    # [T, 4, 3]
    height = feet[:, :, 1] - feet[:, :, 1].min(axis=1, keepdims=True)
    speed = np.zeros(height.shape, dtype=np.float32)
    speed[1:] = np.linalg.norm(np.diff(feet, axis=0), axis=2) * fps
    speed[0] = speed[1]
    w = np.exp(-height / height_scale) * np.exp(-speed / speed_scale)
    return w, speed.min(axis=1)


def lowpass(x: np.ndarray, fps: float, cutoff_hz: float) -> np.ndarray:
    if cutoff_hz is None or cutoff_hz <= 0 or cutoff_hz >= fps / 2:
        return x
    b, a = butter(2, cutoff_hz / (fps / 2), btype="low")
    return filtfilt(b, a, x, axis=0)


def footlock_translation(joints: np.ndarray, lm_transl: np.ndarray, ground_y: float,
                         fps: float = 30.0, stance_floor: float = 1.0,
                         sole_frames: int = 30, cutoff_hz: float = 6.0,
                         guard_m: float | None = None) -> dict:
    """Root translation from the stance foot, falling back to the LM trajectory.

    Args:
        joints: root-frame body joints, [T, 25, 3].
        lm_transl: the LM's own decoded translation, [T, 3] (prior / fallback).
        ground_y: height of the predicted ground plane in the same frame.
    """
    n_time = len(joints)
    lm_delta = np.zeros_like(lm_transl)
    lm_delta[1:] = np.diff(lm_transl, axis=0)

    # Sole offset: the sequences start standing (the window begins after the
    # T-pose), so the first frames calibrate ankle/toe joint height above ground.
    start = joints[:sole_frames, FOOT_JOINTS, 1].min(axis=1)
    sole = float(np.median(start)) + float(lm_transl[:sole_frames, 1].mean()) - ground_y
    ground_offset = ground_y + sole - float(lm_transl[:sole_frames, 1].mean())

    w, slowest = contact_weights(joints, fps)                        # [T, 4], [T]
    feet = joints[:, FOOT_JOINTS]
    d_feet = np.zeros_like(feet)
    d_feet[1:] = np.diff(feet, axis=0)

    total = w.sum(axis=1, keepdims=True)
    # Airborne detection: no foot is slow enough to be planted.
    stance = np.clip((stance_floor - slowest) / stance_floor, 0.0, 1.0)
    w_norm = w / np.clip(total, 1e-6, None)
    # The root moves opposite to the stance foot, which is static in the world.
    v_foot = -(w_norm[:, :, None] * d_feet).sum(axis=1)              # [T, 3]

    # Horizontal: blend the foot-derived velocity with the LM prior by confidence.
    delta = stance[:, None] * v_foot + (1.0 - stance[:, None]) * lm_delta
    transl = np.cumsum(delta, axis=0)
    transl -= transl[0]

    # Vertical: put the lowest foot on the ground while in contact, integrate the
    # LM's vertical delta while airborne (so jumps and sitting still work).
    low = joints[:, FOOT_JOINTS, 1].min(axis=1)
    y_contact = ground_offset - low
    y = np.empty(n_time, dtype=np.float32)
    y[0] = lm_transl[0, 1]
    for t in range(1, n_time):
        y[t] = stance[t] * y_contact[t] + (1.0 - stance[t]) * (y[t - 1] + lm_delta[t, 1])
    transl[:, 1] = y - y[0] + lm_transl[0, 1]

    # --- levitation guard --------------------------------------------------
    # The correction is applied through a gate so the pass can be a strict no-op
    # where the LM trajectory is already physically consistent (which is the case
    # on the synthetic test sets, where the LM traj error is 1-50 mm and the foot
    # lock -- an integrator of the pose error -- is strictly worse).  The gate
    # measures how far the body has left the ground plane relative to the clip's
    # own standing baseline, the one violation that is unambiguous without GT.
    world_low = joints[:, FOOT_JOINTS, 1].min(axis=1) + lm_transl[:, 1]
    base = float(np.median(world_low[:sole_frames] - ground_y))
    deviation = np.abs((world_low - ground_y) - base)
    if guard_m is None:
        gate = np.ones(n_time, dtype=np.float32)
    else:
        gate = np.clip((deviation - guard_m) / max(guard_m, 1e-6), 0.0, 1.0).astype(np.float32)

    # Blend in delta space and low-pass only the correction, so gate == 0 keeps
    # the LM trajectory bit for bit.
    correction = np.zeros_like(lm_transl)
    correction[1:] = np.diff(transl - lm_transl, axis=0)
    correction = lowpass(correction, fps, cutoff_hz)
    transl = lm_transl + np.cumsum(gate[:, None] * correction, axis=0)
    transl = transl - transl[0] + lm_transl[0]
    return {"transl": transl.astype(np.float32), "stance": stance.astype(np.float32),
            "contact_weights": w.astype(np.float32), "sole_offset": sole,
            "gate": gate, "max_ground_deviation": float(deviation.max())}


#: Sources whose IMU is measured rather than simulated.  The foot lock only
#: helps here: on the synthetic test sets the LM trajectory is already at 1-50 mm
#: and the foot lock, being an integrator of the pose error, is strictly worse
#: (measured: 0.012 -> 0.015 m at 60 frames, 0.032 -> 0.066 m at 480).
REAL_IMU_SOURCES = ("imuposer", "dipimu", "ncsa")


_STRING_SETTINGS = {"true": True, "1": True, "yes": True, "on": True,
                    "false": False, "0": False, "no": False, "off": False, "": False}


def enabled_for(source: str, setting) -> bool:
    """``True``/``False`` force the pass; ``"auto"`` restricts it to real IMU.

    Strings are parsed, not just compared against ``"auto"``: a profile that
    reads the switch from the environment gets ``"true"`` / ``"false"``, and the
    old ``else False`` silently turned every non-``auto`` string into "off".
    """
    if isinstance(setting, str):
        value = setting.strip().lower()
        if value == "auto":
            return str(source).lower() in REAL_IMU_SOURCES
        if value not in _STRING_SETTINGS:
            raise ValueError(
                f"experiment.eval_footlock.enabled must be auto/true/false, got {setting!r}"
            )
        return _STRING_SETTINGS[value]
    return bool(setting)


def as_numpy(x) -> np.ndarray:
    """Accept the evaluator's tensors (possibly on the GPU) as well as arrays."""
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()   # .float(): the evaluator runs in bf16
    return np.asarray(x)


def footlock_sample(sample_pred: dict, fps: float | None = None, **kwargs) -> np.ndarray:
    """Corrected root translation for one evaluator ``sample_pred`` dict."""
    pred = sample_pred["pred"]
    joints = forward_kinematics(as_numpy(pred["orient"]), as_numpy(pred["pose"]))
    lm_transl = as_numpy(pred["transl"]).astype(np.float32)
    ground = (pred.get("objects") or {}).get("ground")
    if ground is not None:
        ground_y = float(as_numpy(ground["transl"]).reshape(-1)[1])
    else:
        ground_y = float(lm_transl[:30, 1].mean() + joints[:30, FOOT_JOINTS, 1].min(axis=1).mean())
    if fps is None:
        fps = float(sample_pred.get("metadata", {}).get("fps", 30.0))
    return footlock_translation(joints, lm_transl, ground_y, fps=fps, **kwargs)["transl"]


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    err = pred - gt
    return {
        "traj_l1_m": float(np.abs(err).mean()),
        "traj_l1_per_axis": np.abs(err).mean(axis=0).round(3).tolist(),
        "final_drift_m": float(np.linalg.norm(err[-1])),
        "path_len_m": float(np.linalg.norm(np.diff(pred, axis=0), axis=1).sum()),
        "net_displacement_m": float(np.linalg.norm(pred[-1] - pred[0])),
    }


def process(rollout_path: Path, model_path: str, out_suffix: str, use_gt_pose: bool = False,
            **kwargs) -> dict:
    with rollout_path.open("rb") as f:
        roll = pickle.load(f)
    src = "gt" if use_gt_pose else "pred"
    fps = float(roll["metadata"].get("fps", 30.0))
    joints = forward_kinematics(roll[src]["orient"], roll[src]["pose"], model_path)
    ground = roll["pred"]["objects"].get("ground")
    ground_y = float(ground["transl"][1]) if ground is not None else float(
        roll["pred"]["transl"][:30, 1].mean() + joints[:30, FOOT_JOINTS, 1].min(axis=1).mean())
    out = footlock_translation(joints, roll["pred"]["transl"], ground_y, fps=fps, **kwargs)

    gt = roll["gt"]["transl"]
    result = {
        "session": f"{roll['metadata']['actor_id']}/{roll['metadata']['motion_id']}",
        "before": metrics(roll["pred"]["transl"], gt),
        "after": metrics(out["transl"], gt),
        "gt_path_len_m": float(np.linalg.norm(np.diff(gt, axis=0), axis=1).sum()),
        "gt_net_displacement_m": float(np.linalg.norm(gt[-1] - gt[0])),
        "mean_stance_confidence": float(out["stance"].mean()),
    }
    if out_suffix:
        roll["pred"]["transl"] = out["transl"]
        roll["pred"]["transl_lm"] = np.asarray(result["before"]["traj_l1_m"], dtype=np.float32)
        roll["metadata"]["footlock"] = {"sole_offset": out["sole_offset"], "ground_y": ground_y,
                                        "mean_stance": result["mean_stance_confidence"]}
        dst = rollout_path.with_name(rollout_path.stem + out_suffix + rollout_path.suffix)
        with dst.open("wb") as f:
            pickle.dump(roll, f)
        result["written"] = str(dst)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rollout", nargs="+", required=True)
    ap.add_argument("--smplx-model", default="data/models")
    ap.add_argument("--out-suffix", default="_footlock", help="'' to only print metrics")
    ap.add_argument("--gt-pose", action="store_true", help="drive the foot lock with the GT pose (ceiling)")
    ap.add_argument("--stance-floor", type=float, default=1.0,
                    help="root-frame foot speed (m/s) above which the frame counts as airborne")
    ap.add_argument("--cutoff-hz", type=float, default=6.0)
    ap.add_argument("--guard-m", type=float, default=None,
                    help="only correct frames whose body has left the ground plane by more than "
                         "this (m) relative to the clip's standing baseline; unset = always correct")
    args = ap.parse_args()

    rows = []
    for path in args.rollout:
        rows.append(process(Path(path), args.smplx_model, args.out_suffix, use_gt_pose=args.gt_pose,
                            stance_floor=args.stance_floor, cutoff_hz=args.cutoff_hz,
                            guard_m=args.guard_m))
    header = f"{'session':22s} {'traj L1 before':>14s} {'after':>8s} {'net displ before/after/gt':>27s} {'stance':>7s}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['session']:22s} {r['before']['traj_l1_m']:14.3f} {r['after']['traj_l1_m']:8.3f}"
              f" {r['before']['net_displacement_m']:9.2f} /{r['after']['net_displacement_m']:6.2f} /"
              f"{r['gt_net_displacement_m']:6.2f}      {r['mean_stance_confidence']:5.2f}")
    before = np.mean([r["before"]["traj_l1_m"] for r in rows])
    after = np.mean([r["after"]["traj_l1_m"] for r in rows])
    print(f"{'MEAN':22s} {before:14.3f} {after:8.3f}   ({100 * (after - before) / before:+.0f}%)")


if __name__ == "__main__":
    main()
