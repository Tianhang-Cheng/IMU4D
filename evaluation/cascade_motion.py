"""Feed one run's predicted motion into a motion-input run (i2m -> m2t).

The cascade reuses the per-sample motion a completed benchmark already wrote --
``<metrics>/<layout>/per_sequence/id_<N>_step_<S>.npy`` -- instead of rerunning
the upstream generator. Those files hold the generator's own decisions
(``pred.transl``/``pred.orient``/``pred.pose``, absolute, world frame,
axis-angle, low-pass filtered exactly as the upstream run scored them), so a
downstream profile that would otherwise read clean ground-truth motion reads the
prediction instead and its caption metrics become end-to-end numbers.

Two steps, because the saved dicts also pickle a ``pred.pose_raw`` CUDA tensor
that only unpickles on a machine with the same device visible:

1. ``build`` walks a per_sequence directory once, keeps the three arrays the
   quantizer consumes, converts them to the generator's variable layout
   (``[transl(3) | orient6d(6) | pose6d(126)]``) and packs every sample into one
   flat ``.npz``.
2. ``CascadeMotionSource`` memory-maps that cache and hands run.py's evaluation
   loop a per-sample override keyed by sample id.

Build a cache::

    python -m evaluation.cascade_motion build \\
        exp/ablation_cascade/i2m/evaluation/benchmarks/humanml/step-100000/jobs/\\
frames-480/evaluation/full/step-100000/frames-480/datasets/humanml/metrics/5pt_full \\
        --out exp/ablation_cascade/cascade_inputs/i2m_humanml_5pt_full_f480.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA_VERSION = 1

# What imu_to_input concatenates before quantizing: transl, root orient as 6D,
# and 21 body joints as 6D. Mirrors the ``absolute_traj_absolute_orient6d_pose6d``
# target layout the upstream run records in its token bundle.
TRAJ_NVARS = 3
ORIENT_NVARS = 6
NUM_BODY_JOINTS = 21
POSE_NVARS = NUM_BODY_JOINTS * 6
MOTION_NVARS = TRAJ_NVARS + ORIENT_NVARS + POSE_NVARS


def _sample_id(payload: dict) -> str:
    """The id the upstream run stored, unwrapped from its 1-element array."""
    raw = payload["sample_idx"]
    if isinstance(raw, np.ndarray):
        raw = raw.reshape(-1)
        if raw.size != 1:
            raise ValueError(f"sample_idx holds {raw.size} entries, expected 1")
        return str(raw[0])
    if isinstance(raw, (list, tuple)):
        if len(raw) != 1:
            raise ValueError(f"sample_idx holds {len(raw)} entries, expected 1")
        return str(raw[0])
    return str(raw)


def _to_motion(pred: dict) -> np.ndarray:
    """``pred`` -> ``[T, 135]`` in the generator's variable order.

    ``pred.pose_raw`` is the same rotation already in 6D, but it is the one
    field saved as a CUDA tensor, so the axis-angle arrays are converted here
    instead and the cache stays device-free.
    """
    from utils.rotation2 import convert_rotation
    import torch

    transl = np.ascontiguousarray(pred["transl"], dtype=np.float32)
    orient_aa = np.ascontiguousarray(pred["orient"], dtype=np.float32)
    pose_aa = np.ascontiguousarray(pred["pose"], dtype=np.float32)

    frames = transl.shape[0]
    if transl.shape != (frames, 3):
        raise ValueError(f"pred.transl has shape {transl.shape}, expected (T, 3)")
    if orient_aa.shape != (frames, 3):
        raise ValueError(f"pred.orient has shape {orient_aa.shape}, expected (T, 3)")
    if pose_aa.shape != (frames, NUM_BODY_JOINTS * 3):
        raise ValueError(
            f"pred.pose has shape {pose_aa.shape}, expected (T, {NUM_BODY_JOINTS * 3})"
        )

    orient_6d = convert_rotation(
        torch.from_numpy(orient_aa), "aa", "6d"
    ).reshape(frames, ORIENT_NVARS).numpy()
    pose_6d = convert_rotation(
        torch.from_numpy(pose_aa.reshape(-1, 3)), "aa", "6d"
    ).reshape(frames, POSE_NVARS).numpy()

    return np.concatenate([transl, orient_6d, pose_6d], axis=-1).astype(np.float32)


def build(per_sequence_root: Path, out_path: Path) -> dict:
    """Pack one per_sequence directory into a flat cache and return its metadata."""
    per_sequence = per_sequence_root / "per_sequence"
    if not per_sequence.is_dir():
        # Accept the per_sequence directory itself as well as its parent.
        if per_sequence_root.name == "per_sequence":
            per_sequence = per_sequence_root
        else:
            raise FileNotFoundError(f"No per_sequence directory below {per_sequence_root}")

    paths = sorted(
        per_sequence.glob("id_*.npy"),
        key=lambda p: int(re.search(r"id_(\d+)_", p.name).group(1)),
    )
    if not paths:
        raise FileNotFoundError(f"No id_*.npy under {per_sequence}")

    ids: list[str] = []
    motions: list[np.ndarray] = []
    for index, path in enumerate(paths):
        payload = np.load(path, allow_pickle=True).item()
        pred = payload.get("pred")
        if not isinstance(pred, dict):
            raise ValueError(f"{path} has no 'pred' dict")
        ids.append(_sample_id(payload))
        motions.append(_to_motion(pred))
        if (index + 1) % 250 == 0:
            print(f"  {index + 1}/{len(paths)} samples packed", flush=True)

    if len(set(ids)) != len(ids):
        raise ValueError(f"{per_sequence} contains duplicate sample ids")

    lengths = np.asarray([m.shape[0] for m in motions], dtype=np.int64)
    offsets = np.zeros(len(motions) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        ids=np.asarray(ids),
        lengths=lengths,
        offsets=offsets,
        motion=np.concatenate(motions, axis=0),
        schema_version=np.asarray(SCHEMA_VERSION),
        source=np.asarray(str(per_sequence)),
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "source": str(per_sequence),
        "samples": len(ids),
        "frames_total": int(lengths.sum()),
        "frames_min": int(lengths.min()),
        "frames_max": int(lengths.max()),
        "motion_nvars": MOTION_NVARS,
        "cache": str(out_path),
    }
    out_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


class CascadeMotionSource:
    """Per-sample predicted motion, keyed by sample id.

    ``override`` rewrites the three motion fields of an evaluation batch in
    place. The batch keeps its own tensor device and dtype, so the quantizer,
    the labels and every downstream head see the prediction exactly where they
    would have seen the ground truth.
    """

    def __init__(self, cache_path: str | Path):
        self.path = Path(cache_path)
        with np.load(self.path, allow_pickle=False) as handle:
            schema = int(handle["schema_version"])
            if schema != SCHEMA_VERSION:
                raise ValueError(
                    f"{self.path} has schema {schema}, this build expects {SCHEMA_VERSION}"
                )
            self.ids = [str(v) for v in handle["ids"]]
            self._offsets = handle["offsets"]
            self._motion = handle["motion"]
            self.source = str(handle["source"])
        if self._motion.shape[-1] != MOTION_NVARS:
            raise ValueError(
                f"{self.path} stores {self._motion.shape[-1]} motion variables, "
                f"expected {MOTION_NVARS}"
            )
        self._index = {sample_id: i for i, sample_id in enumerate(self.ids)}
        self.hits = 0
        self.misses: list[str] = []

    def __len__(self) -> int:
        return len(self.ids)

    def motion(self, sample_id: str) -> np.ndarray | None:
        position = self._index.get(str(sample_id))
        if position is None:
            return None
        start, end = int(self._offsets[position]), int(self._offsets[position + 1])
        return self._motion[start:end]

    def override(self, imu_batch: dict) -> bool:
        """Replace ``transl``/``orient``/``pose`` with the cached prediction.

        Returns False when this sample is not in the cache, so the caller can
        decide between skipping it and failing the pass. Raises when the sample
        is present but its length disagrees with the batch -- that means the two
        runs did not evaluate the same window and the numbers would be silently
        misaligned.
        """
        import torch

        sample_id = imu_batch.get("sample_idx")
        if isinstance(sample_id, (list, tuple, np.ndarray)) and len(sample_id) == 1:
            sample_id = sample_id[0]
        sample_id = str(sample_id)

        motion = self.motion(sample_id)
        if motion is None:
            self.misses.append(sample_id)
            return False

        reference = imu_batch["transl"]
        frames = len(reference)
        if motion.shape[0] != frames:
            raise ValueError(
                f"cascade motion for {sample_id} has {motion.shape[0]} frames but the "
                f"evaluation batch has {frames}; the upstream run and this pass did not "
                f"score the same window (cache: {self.path})"
            )

        tensor = torch.from_numpy(np.ascontiguousarray(motion))
        if isinstance(reference, torch.Tensor):
            tensor = tensor.to(device=reference.device, dtype=reference.dtype)
            transl, orient, pose = tensor.split(
                [TRAJ_NVARS, ORIENT_NVARS, POSE_NVARS], dim=-1
            )
        else:
            array = tensor.numpy().astype(reference.dtype, copy=False)
            transl = array[:, :TRAJ_NVARS]
            orient = array[:, TRAJ_NVARS:TRAJ_NVARS + ORIENT_NVARS]
            pose = array[:, TRAJ_NVARS + ORIENT_NVARS:]

        imu_batch["transl"] = transl.contiguous() if hasattr(transl, "contiguous") else transl
        imu_batch["orient"] = orient.contiguous() if hasattr(orient, "contiguous") else orient
        imu_batch["pose"] = pose.contiguous() if hasattr(pose, "contiguous") else pose
        self.hits += 1
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build", help="pack a per_sequence directory into a cascade cache"
    )
    build_parser.add_argument(
        "per_sequence_root",
        help="the layout directory holding per_sequence/, or per_sequence/ itself",
    )
    build_parser.add_argument("--out", required=True, help="destination .npz")

    inspect_parser = subparsers.add_parser("inspect", help="summarise a built cache")
    inspect_parser.add_argument("cache")

    args = parser.parse_args()

    if args.command == "build":
        metadata = build(Path(args.per_sequence_root).resolve(), Path(args.out).resolve())
        print(json.dumps(metadata, indent=2, sort_keys=True))
    else:
        source = CascadeMotionSource(args.cache)
        print(f"{len(source)} samples from {source.source}")
        print(f"first ids: {source.ids[:3]}")


if __name__ == "__main__":
    main()
