"""Classic (streaming) WebDataset loader for IMU4D training.

Drop-in replacement for the map-style ``IMUDataset`` + ``DistributedSampler`` +
``set_cut_length`` combination, for people who want the portable streaming
WebDataset format (see ``pack_wds.py`` for the shard writer).

Design decisions that address the DDP / cut-length risks of streaming wds:

  * **No DDP deadlock**: shards are read with ``resampled=True`` (infinite
    sampling with replacement) and the epoch length is fixed with
    ``with_epoch(steps_per_epoch)``. Every rank therefore produces *exactly* the
    same number of batches, so ``accelerator.gather`` / all-reduce never hangs on
    an uneven shard split. Trade-off: samples are not seen exactly once per epoch.
  * **Per-epoch cut length across worker processes**: ``cut_length`` lives in a
    ``multiprocessing.Value`` captured by the decode closure. On Linux (fork
    start method, which is the default) worker processes share it, so calling
    :func:`set_cut_length_wds` from the main process is visible to the workers on
    the next batch — mirroring the original per-epoch cut behaviour.
    ``SharedCut(per_batch=True)`` switches to one random length per batch
    (:func:`length_bucketed_batches`) for long-context stages where a single
    long per-epoch cut would drop most of the corpus.
  * **Too-short / filtered samples**: the decode step returns ``None`` and a
    ``.select`` drops them; because the stream is infinite + ``with_epoch`` fixed,
    dropping never desynchronises ranks.

The per-sample output dict and the ``process_imu_data`` call are identical to
``IMUDataset.__getitem__`` so the downstream ``imu_to_input`` is unchanged.
Batches are yielded as plain ``list[dict]`` (same as ``collate_fn = return batch``).
"""

import functools
import logging
import multiprocessing as mp
import os
import json
import re
from pathlib import Path
from typing import Any, Callable, Optional

import webdataset as wds

from dataset_process.ncsa.chair_objects import attach_ncsa_objects
from dataset_process.object_taxonomy import normalize_object_annotations

from training.imu_dataset import process_imu_data, process_real_imu_data


def list_wds_shards(root: str, split: str) -> list[str]:
    """Return the absolute shard paths for a split from the packed manifest.

    Args:
        root: The ``imu_data_path`` root (contains ``wds/manifest.json``).
        split: One of ``train`` / ``val`` / ``test``.

    Returns:
        Sorted list of absolute ``.tar`` shard paths.
    """
    import glob

    wds_root = os.path.join(root, "wds")
    manifest_path = os.path.join(wds_root, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        split_info = manifest["splits"][split]
        shared_with = split_info.get("shared_with")
        if shared_with is not None:
            return list_wds_shards(root, shared_with)
        n = split_info["num_shards"]
        return [os.path.join(wds_root, split, f"{split}-{i:06d}.tar") for i in range(n)]
    # Fallback: glob the split directory directly.
    return sorted(glob.glob(os.path.join(wds_root, split, f"{split}-*.tar")))


def num_wds_samples(root: str, split: str) -> int:
    """Return the number of samples in a split from the packed manifest."""
    with open(os.path.join(root, "wds", "manifest.json"), "r") as f:
        manifest = json.load(f)
    return int(manifest["splits"][split]["num_samples"])


def num_wds_contiguous_eval_windows(
    root: str,
    split: str,
    cut_length: int,
    *,
    target_datasets: Optional[list[str]] = None,
    min_clip_length: Optional[int] = None,
    min_window_length: int = 24,
    sample_id_regex: Optional[str] = None,
    sample_ids: Optional[list[str]] = None,
) -> int:
    """Count deterministic non-overlapping evaluation windows from split index.

    Packed real-IMU releases retain one long WDS sample per recorded session.
    Their split index stores every sequence's frame count, so a full evaluation
    can report an exact finite window count before inference.  A final
    tail at least ``min_window_length`` frames is retained.
    """
    if cut_length <= 0:
        raise ValueError(f"cut_length must be positive, got {cut_length}")
    if min_window_length <= 0:
        raise ValueError(
            f"min_window_length must be positive, got {min_window_length}"
        )
    index_path = Path(root) / "splits" / f"{split}.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(
            "Contiguous-window evaluation needs the packed split index: "
            f"{index_path}"
        )
    allowed_prefixes = None if target_datasets is None else set(target_datasets)
    allowed_ids = None if sample_ids is None else set(sample_ids)
    pattern = re.compile(sample_id_regex) if sample_id_regex else None
    minimum = 1 if min_clip_length is None else int(min_clip_length)
    total = 0
    with index_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                sample_id = str(record["id"])
                frames = int(record["frames"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Invalid split index record at {index_path}:{line_number}"
                ) from error
            if allowed_prefixes is not None and sample_id.split("/", 1)[0] not in allowed_prefixes:
                continue
            if allowed_ids is not None and sample_id not in allowed_ids:
                continue
            if pattern is not None and pattern.search(sample_id) is None:
                continue
            if frames < minimum:
                continue
            full_windows, tail = divmod(frames, cut_length)
            total += full_windows
            if tail >= min_window_length:
                total += 1
    if total <= 0:
        raise ValueError(
            f"No tokenizable contiguous {cut_length}-frame windows selected from {index_path}"
        )
    return total


# Per-upstream-prefix sample counts of every split, written next to the
# manifest by ``dataset_process/wds_pipeline/index_prefix_counts.py``.  The
# manifest only knows how many samples a split was packed with, while training
# keeps a subset of them (DEFAULT_TARGET_DATASETS minus
# DEFAULT_EXCLUDED_DATASETS), so epoch accounting needs this breakdown.
PREFIX_COUNTS_FILENAME = "prefix_counts.json"
# ``{shard: {size, samples: [[__key__, id], ...]}}`` per split, same builder.
SAMPLE_INDEX_FILENAME = "sample_index.json"


def load_prefix_counts(root: str, split: str) -> Optional[dict]:
    """Return ``{prefix: count}`` for a split, or ``None`` when unusable.

    ``None`` means the index is absent or stale (its total disagrees with the
    manifest, i.e. the split was repacked after the index was built); callers
    fall back to the manifest total.

    Args:
        root: The ``imu_data_path`` root (contains ``wds/manifest.json``).
        split: One of ``train`` / ``val`` / ``test``.
    """
    path = os.path.join(root, "wds", PREFIX_COUNTS_FILENAME)
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        index = json.load(f)
    entry = index.get("splits", {}).get(split)
    if not entry:
        return None
    prefixes = {str(k): int(v) for k, v in entry.get("prefixes", {}).items()}
    total = num_wds_samples(root, split)
    if sum(prefixes.values()) != total:
        logging.warning(
            "Stale %s for %s/%s (%d indexed vs %d packed); rebuild it with "
            "dataset_process/wds_pipeline/index_prefix_counts.py",
            PREFIX_COUNTS_FILENAME, root, split, sum(prefixes.values()), total,
        )
        return None
    return prefixes


def load_sample_index(root: str, split: str) -> Optional[dict[str, list[tuple[str, str]]]]:
    """Return ``{shard basename: [(__key__, sample id), ...]}`` for a split, or ``None``.

    Built next to the manifest by
    ``dataset_process/wds_pipeline/index_prefix_counts.py``. WDS member names
    are opaque counters (``pack_wds.py``), so without this index the only way
    to find the samples of one source is to unpickle every member of every
    shard -- ~17 GB for the 1.1% of the MotionMillion test split that is LINGO.

    ``None`` means absent or stale: the indexed total disagrees with the
    manifest, or a shard's size on disk differs from the one recorded at
    build time (a repack with the same count would otherwise go unnoticed;
    one that also keeps every shard's byte size is not detected). Callers
    then fall back to the full stream. A split aliased to another
    (manifest ``shared_with``) reads that split's entry.
    """
    wds_root = os.path.join(root, "wds")
    manifest_path = os.path.join(wds_root, "manifest.json")
    path = os.path.join(wds_root, SAMPLE_INDEX_FILENAME)
    if not os.path.isfile(path) or not os.path.isfile(manifest_path):
        return None
    with open(manifest_path, "r") as f:
        split_info = json.load(f)["splits"].get(split)
    if not split_info:
        return None
    stored_split = split_info.get("shared_with") or split
    with open(path, "r") as f:
        entry = json.load(f).get("splits", {}).get(stored_split)
    if not entry:
        return None
    shards = {}
    for shard, shard_entry in entry.get("shards", {}).items():
        shard_path = os.path.join(wds_root, stored_split, str(shard))
        if not os.path.isfile(shard_path) or os.path.getsize(shard_path) != int(shard_entry["size"]):
            logging.warning(
                "Stale %s for %s/%s: %s changed on disk since indexing; rebuild it with "
                "dataset_process/wds_pipeline/index_prefix_counts.py",
                SAMPLE_INDEX_FILENAME, root, split, shard,
            )
            return None
        shards[str(shard)] = [(str(key), str(sample_id)) for key, sample_id in shard_entry["samples"]]
    total = sum(len(rows) for rows in shards.values())
    if total != num_wds_samples(root, stored_split):
        logging.warning(
            "Stale %s for %s/%s (%d indexed vs %d packed); rebuild it with "
            "dataset_process/wds_pipeline/index_prefix_counts.py",
            SAMPLE_INDEX_FILENAME, root, split, total, num_wds_samples(root, stored_split),
        )
        return None
    return shards


def restrict_eval_shards(
    root: str,
    split: str,
    shards: list[str],
    *,
    target_datasets: Optional[list] = None,
    sample_id_regex: Optional[str] = None,
    sample_ids: Optional[list[str]] = None,
) -> tuple[list[str], Optional[set[str]]]:
    """Drop the shards and members the evaluation filters would discard anyway.

    Returns ``(shards, allowed_keys)``: the shards holding at least one sample
    that passes the prefix / deny-list / regex / id filters of
    :func:`build_eval_wds_loader`, and the ``__key__`` set of those samples so
    the loader can skip unpickling everything else (the key is available
    before ``.decode()``, the id only after). Without a usable
    :func:`load_sample_index` both come back unchanged and ``allowed_keys`` is
    ``None``: the loader then filters after decoding, as before.

    The rules here mirror ``_make_decode_process_fn`` so the decode step keeps
    the final say; this only removes work, never admits a sample.
    """
    index = load_sample_index(root, split)
    if index is None:
        return list(shards), None
    allowed_prefixes = set(target_datasets) if target_datasets is not None else None
    excluded_prefixes = set(DEFAULT_EXCLUDED_DATASETS) - (allowed_prefixes or set())
    pattern = re.compile(sample_id_regex) if sample_id_regex else None
    allowed_ids = None if sample_ids is None else set(sample_ids)

    def _selected(sample_id: str) -> bool:
        prefix = sample_id.split("/")[0]
        if allowed_prefixes is not None and prefix not in allowed_prefixes:
            return False
        if prefix in excluded_prefixes:
            return False
        if pattern is not None and pattern.search(sample_id) is None:
            return False
        return allowed_ids is None or sample_id in allowed_ids

    kept_shards: list[str] = []
    allowed_keys: set[str] = set()
    for shard in shards:
        rows = index.get(os.path.basename(str(shard)))
        if rows is None:
            # A shard the index does not know: filtering keys would silently
            # drop its samples, so do not filter at all.
            return list(shards), None
        keys = [key for key, sample_id in rows if _selected(sample_id)]
        if keys:
            kept_shards.append(shard)
            allowed_keys.update(keys)
    if not kept_shards:
        raise ValueError(
            f"No sample in {root} [{split}] passes the evaluation filters "
            f"(target_datasets={target_datasets}, sample_id_regex={sample_id_regex!r}, "
            f"{0 if sample_ids is None else len(sample_ids)} exact ids)"
        )
    return kept_shards, allowed_keys


def num_wds_trainable_samples(
    root: str,
    split: str,
    target_datasets: Optional[list] = None,
) -> tuple[int, bool]:
    """Return how many samples of a split the loader's filters actually keep.

    The decode step drops every sample whose id prefix is outside
    ``target_datasets`` (default :data:`DEFAULT_TARGET_DATASETS`) or inside
    :data:`DEFAULT_EXCLUDED_DATASETS`; for MotionMillion that is most of the
    split, so the packed count badly overstates one pass over the data.

    Args:
        root: The ``imu_data_path`` root.
        split: One of ``train`` / ``val`` / ``test``.
        target_datasets: Allow-list, matching ``build_train_wds_loader``.

    Returns:
        ``(count, exact)``.  ``exact`` is False when no usable prefix index was
        found, in which case ``count`` is the packed total (an upper bound).
    """
    total = num_wds_samples(root, split)
    prefixes = load_prefix_counts(root, split)
    if prefixes is None:
        return total, False
    allowed = set(DEFAULT_TARGET_DATASETS if target_datasets is None else target_datasets)
    excluded = set(DEFAULT_EXCLUDED_DATASETS) - allowed
    kept = sum(
        count
        for prefix, count in prefixes.items()
        if prefix in allowed and prefix not in excluded
    )
    if kept <= 0:
        logging.warning(
            "No sample of %s/%s matches the dataset allow-list %s; falling back "
            "to the packed total for epoch accounting.",
            root, split, sorted(allowed),
        )
        return total, False
    return kept, True

# Which process_* branch each shard source maps to, mirroring the sample_idx
# prefix routing in IMUDataset.__getitem__.
_SOURCE_TO_DATA_SOURCE = {
    "motionmillion": "other_dataset",
    "hiphi": "other_dataset",
    "omomo": "other_dataset",
    "humoto": "humoto",
    # Real-world IMU releases: measured readings, no virtual-IMU simulation.
    "imuposer": "imuposer",
    "dipimu": "dipimu",
    "ncsa": "ncsa",
}

# Shard sources whose samples carry measured ``imu_acc`` / ``imu_ori`` instead
# of virtual-sensor trajectories (``imu_traj``); routed to process_real_imu_data.
REAL_IMU_SOURCES = frozenset({"imuposer", "dipimu", "ncsa"})

# These releases describe human motion and/or the interacted object, but do not
# carry a literal floor entry in each sample.  The map-style loader has always
# supplied a synthetic ground plane for them. MotionMillion / DIP need a
# per-clip estimate: their storage frame has no floor-at-zero guarantee.
_SYNTHETIC_GROUND_SOURCES = frozenset({"motionmillion", "hiphi", "omomo", "imuposer", "dipimu", "ncsa"})


def needs_synthetic_ground(source: str, objects: dict) -> bool:
    """Return whether the loader should supply a missing ground object."""

    return source in _SYNTHETIC_GROUND_SOURCES and "ground" not in objects


# Default dataset allow-list, matching ``target_datasets`` in IMUDataset.__init__.
# NOTE: MotionGV / Mirror_MotionGV are intentionally EXCLUDED (flagged "too noisy,
# harmful for training" in imu_dataset.py); pass an explicit list to override.
DEFAULT_TARGET_DATASETS = [
    "LINGO",
    "BABEL",
    "Mirror_BABEL",
    "PhantomDanceDatav1.1",
    "Mirror_PhantomDanceDatav1.1",
    "MotionLLAMA",
    "MotionUnion",
    "Mirror_MotionLLAMA",
    "Mirror_MotionUnion",
    "HiPHI",
    "OMOMO",
    "HUMOTO",
]

# Datasets dropped from every split unless a caller names them explicitly in
# ``target_datasets``.  MotionGV is recovered from monocular video: many of its
# clips barely move and their captions describe actions the recovered pose never
# performs (measured 2026-09-10: median pelvis-relative travel 62 mm/s vs 114+
# for every kept subset; 37% of its clips whose text promises vigorous motion
# fall below the mocap 25th percentile).  Training already excluded it via
# DEFAULT_TARGET_DATASETS, but evaluation did not, leaving ~35% of the val/test
# stream on data the model never trains on.
DEFAULT_EXCLUDED_DATASETS = frozenset({"MotionGV", "Mirror_MotionGV"})

REWRITE_DIRNAME = "qwen3_0.6B_rewrite_v1"
EVAL_REWRITE_DIRNAME = "qwen3_0.6B_rewrite_v2"
REWRITE_OUTPUT_KEY = "qwen3-0.6B"


def default_rewrite_root(wds_root: str) -> str:
    """Return the conventional rewrite-sidecar root for one packed WDS root."""
    return str(Path(wds_root).parent / REWRITE_DIRNAME)


def load_rewrite_texts(
    wds_roots: list[str],
    split: str,
    *,
    rewrite_roots: Optional[list[str]] = None,
    enabled: bool = True,
) -> dict[str, list[str]]:
    """Load valid rewrite sidecars keyed by sample id.

    Sidecars stay outside immutable WebDataset tar files. When ``rewrite_roots``
    is omitted, validation/test use sibling ``qwen3_0.6B_rewrite_v2``;
    training uses ``qwen3_0.6B_rewrite_v1``
    directory. A logical split that shares its WDS shards with another split
    (for example HiPHI/OMOMO/HUMOTO ``test -> val``) reads that physical split's
    rewrite sidecars as well.
    """
    if not enabled:
        return {}
    if rewrite_roots is None:
        rewrite_roots = [
            str(Path(root).parent / EVAL_REWRITE_DIRNAME)
            if split in {"val", "test"} else default_rewrite_root(root)
            for root in wds_roots
        ]
    if len(rewrite_roots) != len(wds_roots):
        raise ValueError(
            "rewrite_roots must have one entry per WDS root, or be omitted for "
            "automatic sibling-directory discovery."
        )

    rewritten: dict[str, list[str]] = {}
    for wds_root, rewrite_root in zip(wds_roots, rewrite_roots, strict=True):
        if not rewrite_root:
            continue
        physical_split = split
        manifest_path = Path(wds_root) / "wds" / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            seen_splits = set()
            while True:
                if physical_split in seen_splits:
                    raise ValueError(
                        f"Cyclic shared_with chain in {manifest_path}: "
                        f"{sorted(seen_splits)}"
                    )
                seen_splits.add(physical_split)
                split_info = manifest.get("splits", {}).get(physical_split, {})
                shared_with = split_info.get("shared_with")
                if shared_with is None:
                    break
                physical_split = str(shared_with)
        sidecar_dir = Path(rewrite_root) / physical_split
        if not sidecar_dir.is_dir():
            continue
        for sidecar_path in sorted(sidecar_dir.glob("*.jsonl")):
            with sidecar_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"Invalid rewrite JSON at {sidecar_path}:{line_number}"
                        ) from error
                    value = record.get(REWRITE_OUTPUT_KEY)
                    caption = (
                        value[0]
                        if isinstance(value, list) and len(value) == 1
                        else None
                    )
                    sample_id = record.get("id")
                    if (
                        record.get("status") != "rewritten"
                        or not isinstance(sample_id, str)
                        or not isinstance(caption, str)
                    ):
                        continue
                    caption = " ".join(caption.split())
                    if not caption:
                        continue
                    previous = rewritten.get(sample_id)
                    if previous is not None and previous != [caption]:
                        raise ValueError(
                            f"Conflicting rewrite captions for {sample_id!r}; "
                            f"found while loading {sidecar_path} for {wds_root}."
                        )
                    rewritten[sample_id] = [caption]
    return rewritten


class SharedCut:
    """Process-shared cut-length state (visible to fork-started DataLoader workers).

    Two modes:

    * **epoch** (``per_batch=False``, historical behaviour): the main process
      publishes one cut length per epoch through :func:`set_cut_length_wds` and
      every sample in that epoch uses it. Sequences shorter than the cut are
      dropped.
    * **per-batch** (``per_batch=True``): each worker draws an independent
      target length in ``[min_length, max_length]`` (a multiple of ``step``,
      i.e. one of a few length buckets) for every batch, keeps only
      clips at least that long (cropped to it) until the batch is full, then
      redraws. The batch size follows a token budget (:meth:`batch_size_for`):
      the configured batch size applies to ``max_length`` and shorter buckets
      take proportionally more samples. Long clips are therefore trained at many lengths, short clips
      still contribute to short-cut batches, and every batch stays temporally
      uniform, which the model's loss code requires (it stacks per-sample
      motion targets). Use this for long-context stages where a single long
      per-epoch cut would drop most of the corpus.
    """

    def __init__(
        self,
        init: int,
        *,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        per_batch: bool = False,
        step: int = 1,
        batch_overhead_frames: int = 0,
    ) -> None:
        """Initialise the shared state.

        Args:
            init: Initial epoch-mode cut length (timesteps).
            min_length: Lower bound of the per-batch range (timesteps).
            max_length: Upper bound of the per-batch range (timesteps).
            per_batch: Draw one random length per batch instead of per epoch.
            step: Bucket size for per-batch draws; lengths are multiples of it
                (``step=60`` with ``[60, 480]`` gives 8 buckets). Few distinct
                shapes keep cudnn autotuning and the CUDA allocator warm.
            batch_overhead_frames: Per-sample fixed token cost expressed in
                frames (text, object and special tokens; ~100 tokens at ~1.75
                tokens/frame is ~60 frames). :meth:`batch_size_for` scales the
                reference batch size by ``(max + overhead) / (L + overhead)`` so
                every bucket's micro-batch holds a similar token count.
        """
        self._v = mp.Value("i", int(init))
        self._step = int(step)
        self._overhead = int(batch_overhead_frames)
        if self._overhead < 0:
            raise ValueError(f"batch_overhead_frames must be >= 0, got {batch_overhead_frames}")
        if self._step < 1:
            raise ValueError(f"step must be >= 1, got {step}")
        self._min = mp.Value("i", int(init if min_length is None else min_length))
        self._max = mp.Value("i", int(init if max_length is None else max_length))
        self._per_batch = bool(per_batch)
        if self._min.value > self._max.value:
            raise ValueError(
                f"min_length ({self._min.value}) must not exceed max_length ({self._max.value})"
            )
        if self._per_batch and not self.buckets():
            raise ValueError(
                f"no multiple of step={self._step} in [{self._min.value}, {self._max.value}]"
            )

    @property
    def per_batch(self) -> bool:
        """Whether the cut length is drawn independently for every batch."""
        return self._per_batch

    def get(self) -> int:
        """Return the current epoch-mode cut length."""
        return self._v.value

    def set(self, value: int) -> None:
        """Update the epoch-mode cut length."""
        self._v.value = int(value)

    def set_range(self, min_length: int, max_length: int) -> None:
        """Update the per-batch range."""
        if int(min_length) > int(max_length):
            raise ValueError(f"min_length ({min_length}) must not exceed max_length ({max_length})")
        self._min.value = int(min_length)
        self._max.value = int(max_length)

    def buckets(self) -> list[int]:
        """Allowed per-batch lengths: multiples of ``step`` in ``[min, max]``."""
        lo, hi, st = self._min.value, self._max.value, self._step
        first = -(-lo // st) * st  # ceil to a multiple of step
        return list(range(first, hi + 1, st))

    def draw(self, rng=None) -> int:
        """Draw a per-batch cut length uniformly from :meth:`buckets`.

        Args:
            rng: Optional ``random.Random`` (defaults to the module RNG, which
                PyTorch seeds per DataLoader worker).
        """
        import random

        return (rng or random).choice(self.buckets())

    def batch_size_for(self, cut_length: int, ref_batch_size: int) -> int:
        """Token-budget batch size for ``cut_length``.

        ``ref_batch_size`` is the batch size at ``max_length``; shorter buckets
        get proportionally more samples (at least 1) so each micro-batch costs
        about the same number of tokens. Epoch mode returns ``ref_batch_size``.
        """
        if not self._per_batch:
            return int(ref_batch_size)
        ov = self._overhead
        scale = (self._max.value + ov) / (int(cut_length) + ov)
        return max(1, int(round(ref_batch_size * scale)))

    def mean_batch_size(self, ref_batch_size: int) -> float:
        """Expected micro-batch size over the per-batch length buckets.

        :meth:`draw` picks a bucket uniformly, so a micro-batch holds on
        average the mean of :meth:`batch_size_for` over :meth:`buckets` samples
        -- roughly twice ``ref_batch_size`` with the default 60..480 range,
        since short buckets take proportionally more samples.  Epoch mode
        returns ``ref_batch_size``.
        """
        if not self._per_batch:
            return float(ref_batch_size)
        buckets = self.buckets()
        return sum(self.batch_size_for(L, ref_batch_size) for L in buckets) / len(buckets)

    def cut_length_for(self, n_time: int) -> Optional[int]:
        """Epoch mode: the published cut length, or ``None`` if the clip is shorter.

        Mirrors ``process_imu_data``'s own ``n_time < cut_length`` check so the
        decode step can drop too-short clips before any heavy processing.
        """
        cut = self._v.value
        return cut if n_time >= cut else None


def set_cut_length_wds(shared_cut: SharedCut, cut_min_length: int, cut_max_length: int) -> None:
    """Publish the epoch's cut length (or the per-batch range) to all workers.

    In epoch mode this picks one random cut length in ``[min, max]``, the
    streaming analogue of ``training.imu_dataset.set_cut_length``. In per-batch
    mode it only refreshes the range; each worker draws its own length per
    batch via :meth:`SharedCut.draw`.

    Args:
        shared_cut: The shared cut holder passed into :func:`build_train_wds_loader`.
        cut_min_length: Minimum cut length (timesteps).
        cut_max_length: Maximum cut length (timesteps).
    """
    if shared_cut.per_batch:
        shared_cut.set_range(cut_min_length, cut_max_length)
        return
    import random

    shared_cut.set(random.randint(cut_min_length, cut_max_length))


def length_bucketed_batches(
    source,
    batch_size: int,
    shared_cut: SharedCut,
    decode_fn: Callable[[dict, int], Optional[dict]],
    rng=None,
):
    """Per-batch random crop length pipeline stage (see :class:`SharedCut`).

    Draws ``L = shared_cut.draw()``, then consumes decoded (but not yet
    processed) samples: clips shorter than ``L`` are skipped cheaply, others are
    processed with ``decode_fn(sample, L)`` and appended. When
    ``shared_cut.batch_size_for(L, batch_size)`` samples are collected the
    batch is yielded as ``list[dict]`` and a new ``L`` is drawn. Samples
    ``decode_fn`` rejects (``None``) are dropped.

    Args:
        source: Iterator of decoded wds samples (``sample["sample.pkl"]`` is
            the unpickled payload).
        batch_size: Reference batch size at ``shared_cut``'s max length
            (never partial batches).
        shared_cut: Range holder in per-batch mode.
        decode_fn: ``_make_decode_process_fn`` closure; called with an explicit
            ``cut_length``.
        rng: Optional ``random.Random`` for tests.
    """
    batch: list = []
    cut_length = shared_cut.draw(rng)
    target = shared_cut.batch_size_for(cut_length, batch_size)
    for sample in source:
        n_time = len(sample["sample.pkl"]["motion_data_smpl85"])
        if n_time < cut_length:
            continue
        out = decode_fn(sample, cut_length)
        if out is None:
            continue
        batch.append(out)
        if len(batch) == target:
            yield batch
            batch = []
            cut_length = shared_cut.draw(rng)
            target = shared_cut.batch_size_for(cut_length, batch_size)


def _make_decode_process_fn(
    shared_cut: SharedCut,
    split: str,
    random_cut: bool,
    random_mask_text: bool,
    shift: int,
    filter_short_text: bool,
    require_text: bool,
    motion_only: bool,
    scene_only: bool,
    dynamic_object: bool,
    fps: int,
    imu_seq_max_len: int,
    add_imu_noise: bool,
    acc_scale: float,
    gyro_scale: float,
    allowed_prefixes: Optional[set] = None,
    excluded_prefixes: Optional[set] = None,
    rewritten_texts: Optional[dict[str, list[str]]] = None,
    min_clip_length: Optional[int] = None,
    sample_id_regex: Optional[str] = None,
    allowed_sample_ids: Optional[set[str]] = None,
    require_rewritten_text: bool = False,
    imu_noise_cfg: Optional[dict] = None,
    smooth_real_imu_acc: bool = False,
    real_heading_aug: Optional[dict] = None,
    short_window_global_supervision: Optional[dict] = None,
) -> Callable[[dict], Optional[dict]]:
    """Build the per-sample decode+process closure used inside the wds pipeline.

    Args:
        allowed_prefixes: If set, only keep samples whose id's top-level dataset
            token (``id.split('/')[0]``) is in this set — mirrors the
            ``target_datasets`` filtering in ``IMUDataset``. ``None`` keeps all.
        excluded_prefixes: Dataset tokens to drop even when ``allowed_prefixes``
            would keep them (see :data:`DEFAULT_EXCLUDED_DATASETS`). Callers
            subtract anything they asked for by name, so an explicit
            ``target_datasets=["MotionGV"]`` still works for debugging.
        min_clip_length: Eval-only shortest clip (frames) to keep when the
            closure is called without an explicit ``cut_length``. ``None``
            (historical) drops every clip shorter than the published cut length;
            with a value, clips of at least that length are kept and windowed to
            the first ``min(len, cut_length)`` frames by ``process_imu_data``.
        sample_id_regex: Optional regular expression applied with ``search`` to
            the complete sample id after the top-level dataset filter. This is
            useful for packed roots containing nested subsets such as
            ``MotionUnion/humanml/...``.
        allowed_sample_ids: Optional exact-id allow-list applied after the
            prefix and regex filters. This pins evaluation to a reusable sample
            manifest instead of relying on the current shard iteration order.
        require_rewritten_text: Drop samples missing from ``rewritten_texts``.
            Evaluation uses this to guarantee that raw WDS captions can never
            silently become references when rewrite supervision is requested.
        imu_noise_cfg: ``training.imu_noise`` mapping (imu_synthesis/imu_noise.py)
            for the synthetic sources; real sensors are never augmented.
        real_heading_aug: ``training.real_imu_heading_aug`` mapping.  Randomises
            the per-device heading conjugation of *measured* readings on the train
            split (training.imu_dataset.apply_real_heading_aug), so a fine-tune
            cannot memorise each session's fixed miscalibration.
        short_window_global_supervision: ``training.short_window_global_supervision``
            mapping. On the train split it restores the world heading / trajectory
            target for crops no longer than ``max_frames`` even when the sample's
            ``motion_supervise`` masks them (training.imu_dataset).
        smooth_real_imu_acc: Apply the historical 3-tap moving average to real
            accelerations that were not smoothed at conversion time.

    Returns:
        A function mapping a decoded wds sample to a processed sample dict, or
        ``None`` when the sequence is filtered out (wrong dataset / too short).
    """

    sample_id_pattern = re.compile(sample_id_regex) if sample_id_regex else None

    def _fn(sample: dict, cut_length: Optional[int] = None) -> Optional[dict]:
        payload = sample["sample.pkl"]  # already unpickled by .decode()
        seq_id = payload["id"]
        window_start = int(sample.get("__imu4d_eval_window_start", 0))
        window_end = sample.get("__imu4d_eval_window_end")
        effective_shift = shift + window_start
        seq_prefix = seq_id.split("/")[0]
        if allowed_prefixes is not None and seq_prefix not in allowed_prefixes:
            return None
        if excluded_prefixes and seq_prefix in excluded_prefixes:
            return None
        if sample_id_pattern is not None and sample_id_pattern.search(seq_id) is None:
            return None
        if allowed_sample_ids is not None and seq_id not in allowed_sample_ids:
            return None
        if require_rewritten_text and seq_id not in (rewritten_texts or {}):
            return None
        source = payload.get("source", "motionmillion")
        data_source = _SOURCE_TO_DATA_SOURCE[source]
        real_imu = source in REAL_IMU_SOURCES

        model_sample = {
            "source": source,
            "id": seq_id,
            "__url__": sample.get("__url__"),
            "motion_data_smpl85": payload["motion_data_smpl85"],
            "texts": (rewritten_texts or {}).get(
                seq_id, payload.get("texts", payload.get("description"))
            ),
            "objects": payload.get("objects", {}),
        }
        if real_imu:
            # Measured readings (dataset_process/realworld): no imu_traj.
            model_sample["imu_acc"] = payload["imu_acc"]
            model_sample["imu_ori"] = payload["imu_ori"]
            model_sample["imu_acc_smoothed"] = bool(payload.get("imu_acc_smoothed", False))
            # False -> process_real_imu_data skips the historical (0,-9.8,0) offset
            model_sample["imu_acc_add_gravity"] = bool(payload.get("imu_acc_add_gravity", True))
        else:
            model_sample["imu_traj"] = payload["imu_traj"]
        for optional_key in (
            "object_metadata",
            "object_valid_mask",
            "object_motion_mask",
            "task_mode",
            "annotation_scope",
            "ground_plane",
            "object_anchor_valid",
            "imu_position_rest_pelvis_correction_applied",
            "world_geometry_rest_pelvis_correction_applied",
            "legacy_rest_pelvis_correction",
            # Per-sample motion-label reliability: which channel groups carry a loss
            # ({"traj": bool, "orient": bool, "pose": bool}); absent = all supervised.
            "motion_supervise",
        ):
            if optional_key in payload:
                model_sample[optional_key] = payload[optional_key]
        if source == "ncsa":
            # Meeting-room chairs from the standalone ncsa/objects release;
            # a no-op unless dataset.params.ncsa_object_root is set.
            attach_ncsa_objects(model_sample, payload)
        (
            model_sample["objects"],
            model_sample["object_metadata"],
            model_sample["object_valid_mask"],
            model_sample["object_motion_mask"],
        ) = normalize_object_annotations(
            model_sample["objects"],
            model_sample.get("object_metadata"),
            model_sample.get("object_valid_mask"),
            model_sample.get("object_motion_mask"),
            source=source,
        )
        if cut_length is None:  # epoch mode: use the published per-epoch cut
            n_time = len(model_sample["motion_data_smpl85"])
            if min_clip_length is not None:
                # Eval with a fixed window: keep every clip at least
                # min_clip_length long; shorter-than-window clips are scored whole.
                if n_time < int(min_clip_length):
                    return None
                cut_length = shared_cut.get()
            else:
                cut_length = shared_cut.cut_length_for(n_time)
                if cut_length is None:
                    return None  # shorter than this epoch's cut length
        if real_imu:
            # Real sensors already carry noise; add_imu_noise / scene_only do
            # not apply.  Mirrors the historical map-style imuposer/dipimu path.
            out = process_real_imu_data(
                model_sample,
                random_cut,
                random_mask_text,
                cut_length,
                shift=effective_shift,
                add_ground_data=needs_synthetic_ground(source, model_sample["objects"]),
                filter_short_text=filter_short_text,
                data_source=data_source,
                dynamic_object=dynamic_object,
                fps=fps,
                split=split,
                sample_idx=seq_id,
                IMUSEQMAXLEN=imu_seq_max_len,
                acc_scale=acc_scale,
                gyro_scale=gyro_scale,
                smooth_imu_acc=smooth_real_imu_acc,
                motion_only=motion_only,
                heading_aug=real_heading_aug,
                short_window_global_supervision=short_window_global_supervision,
            )
        else:
            out = process_imu_data(
                model_sample,
                random_cut,
                random_mask_text,
                cut_length,
                shift=effective_shift,
                add_ground_data=needs_synthetic_ground(source, model_sample["objects"]),
                filter_short_text=filter_short_text,
                motion_only=motion_only,
                scene_only=scene_only,
                data_source=data_source,
                dynamic_object=dynamic_object,
                fps=fps,
                split=split,
                IMUSEQMAXLEN=imu_seq_max_len,
                add_imu_noise=add_imu_noise,
                acc_scale=acc_scale,
                gyro_scale=gyro_scale,
                imu_noise_cfg=imu_noise_cfg,
            )
        if out is None:
            return None
        if require_text and not out.get("description"):
            return None
        if window_end is None:
            out["sample_idx"] = payload["id"]
        else:
            out["sample_idx"] = (
                f"{payload['id']}#frames-{window_start:06d}-{int(window_end):06d}"
            )
        # Lightweight provenance used by structured full-eval artifacts.  Keep
        # it outside process_imu_data so the tensor training interface remains
        # unchanged.
        out["source"] = source
        out["motion_id"] = payload.get("motion_id", str(payload["id"]).split("/")[-1])
        out["actor_id"] = payload.get("actor_id", "unknown")
        out["fps"] = float(payload.get("fps", fps))
        return out

    return _fn


def build_train_wds_loader(
    shards: Any,
    batch_size: int,
    steps_per_epoch: int,
    shared_cut: SharedCut,
    *,
    num_workers: int = 1,
    shuffle_buffer: int = 4000,
    persistent_workers: bool = True,
    random_cut: bool = True,
    filter_short_text: bool = True,
    require_text: bool = False,
    target_datasets: Optional[list] = None,
    motion_only: bool = False,
    scene_only: bool = False,
    dynamic_object: bool = False,
    fps: int = 30,
    imu_seq_max_len: int = 200,
    add_imu_noise: bool = False,
    acc_scale: float = 1.0,
    gyro_scale: float = 1.0,
    rewritten_texts: Optional[dict[str, list[str]]] = None,
    imu_noise_cfg: Optional[dict] = None,
    smooth_real_imu_acc: bool = False,
    real_heading_aug: Optional[dict] = None,
    short_window_global_supervision: Optional[dict] = None,
) -> wds.WebLoader:
    """Build a DDP-safe streaming training loader yielding ``list[dict]`` batches.

    Args:
        shards: A brace-expanded shard URL / list of shard paths for the split.
        batch_size: Number of samples per yielded list-batch.
        steps_per_epoch: Fixed number of batches per epoch (must be identical on
            every rank to keep DDP in sync).
        shared_cut: Shared cut-length holder; update via :func:`set_cut_length_wds`.
        num_workers: DataLoader worker processes.
        shuffle_buffer: Sample-level shuffle buffer size (on top of shard shuffle).
        persistent_workers: Keep workers alive across epochs.
        random_cut: Whether ``process_imu_data`` takes a random contiguous window.
        filter_short_text: Drop samples with too-short/no descriptions.
        require_text: Skip samples that end up without any caption. A third of
            the MotionMillion clips carry none, and the length buckets group
            them into whole caption-less batches; a captioning-only profile
            (model.text_only) gets no supervision from them at all.
        target_datasets: Dataset allow-list (by id top-level token). ``None`` uses
            :data:`DEFAULT_TARGET_DATASETS` (which excludes MotionGV); pass an
            explicit list (e.g. ``["LINGO"]``) to restrict, or a superset to widen.
        motion_only / scene_only / dynamic_object / fps / imu_seq_max_len /
        add_imu_noise / acc_scale / gyro_scale: Passed straight to
            ``process_imu_data`` (same meaning as in ``IMUDataset``).
        rewritten_texts: Optional ``sample id -> [caption]`` mapping from
            sidecars. Missing ids retain their original WDS captions.
        imu_noise_cfg / smooth_real_imu_acc: See :func:`_make_decode_process_fn`.

    Returns:
        A ``webdataset.WebLoader`` whose iteration yields ``list[dict]`` batches.
    """
    allowed_prefixes = set(DEFAULT_TARGET_DATASETS if target_datasets is None else target_datasets)
    excluded_prefixes = set(DEFAULT_EXCLUDED_DATASETS) - allowed_prefixes
    decode_fn = _make_decode_process_fn(
        shared_cut=shared_cut,
        split="train",
        random_cut=random_cut,
        random_mask_text=False,
        shift=0,
        filter_short_text=filter_short_text,
        require_text=require_text and not (motion_only or scene_only),
        motion_only=motion_only,
        scene_only=scene_only,
        dynamic_object=dynamic_object,
        fps=fps,
        imu_seq_max_len=imu_seq_max_len,
        add_imu_noise=add_imu_noise,
        acc_scale=acc_scale,
        gyro_scale=gyro_scale,
        allowed_prefixes=allowed_prefixes,
        excluded_prefixes=excluded_prefixes,
        rewritten_texts=rewritten_texts,
        imu_noise_cfg=imu_noise_cfg,
        smooth_real_imu_acc=smooth_real_imu_acc,
        real_heading_aug=real_heading_aug,
        short_window_global_supervision=short_window_global_supervision,
    )

    dataset = (
        wds.WebDataset(
            shards,
            resampled=True,                 # infinite sampling -> no uneven-shard DDP deadlock
            shardshuffle=False,             # ignored under resampled; resampling already randomizes shards
            nodesplitter=wds.split_by_node,  # shard subset per DDP rank
            workersplitter=wds.split_by_worker,
        )
        .shuffle(shuffle_buffer)
        .decode()
    )
    if shared_cut.per_batch:
        # One random crop length per batch (see SharedCut); batches stay uniform.
        dataset = dataset.compose(
            functools.partial(
                length_bucketed_batches,
                batch_size=batch_size,
                shared_cut=shared_cut,
                decode_fn=decode_fn,
            )
        )
    else:
        dataset = (
            dataset.map(decode_fn)
            .select(lambda x: x is not None)
            .batched(batch_size, collation_fn=lambda batch: list(batch), partial=False)
        )

    loader = wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )
    # Fix the number of batches per epoch so every rank steps in lock-step.
    loader = loader.with_epoch(steps_per_epoch)
    return loader


def build_eval_wds_loader(
    shards: Any,
    cut_length: int,
    *,
    num_workers: int = 1,
    shift: int = 0,
    target_datasets: Optional[list] = None,
    split: str = "test",
    fps: int = 30,
    imu_seq_max_len: int = 200,
    motion_only: bool = False,
    scene_only: bool = False,
    dynamic_object: bool = False,
    rewritten_texts: Optional[dict[str, list[str]]] = None,
    min_clip_length: Optional[int] = None,
    sample_id_regex: Optional[str] = None,
    sample_ids: Optional[list[str]] = None,
    require_rewritten_text: bool = False,
    imu_noise_cfg: Optional[dict] = None,
    smooth_real_imu_acc: bool = False,
    real_heading_aug: Optional[dict] = None,
    short_window_global_supervision: Optional[dict] = None,
    contiguous_windows: bool = False,
    contiguous_window_min_frames: int = 24,
    allowed_keys: Optional[set[str]] = None,
) -> wds.WebLoader:
    """Build a deterministic single-pass eval loader (batch_size=1, no resampling).

    Every distributed rank receives the same ordered stream; ``eval_model``
    partitions sample indices across ranks and gathers metrics. This avoids DDP
    shard-boundary duplication while retaining deterministic full coverage.

    Args:
        shards: Shard URL / list for the eval split.
        cut_length: Fixed cut length for evaluation windows.
        num_workers: DataLoader worker processes.
        shift: Frame shift applied to the cut window (test mode uses 0 and 2).
        target_datasets: Dataset allow-list (by id top-level token). ``None`` keeps
            every dataset in the shards (use e.g. ``["LINGO"]`` for a single-dataset
            benchmark) except :data:`DEFAULT_EXCLUDED_DATASETS`, which is dropped
            from eval as well as training unless named here explicitly. Note the
            allow-list default cannot be used here: real-IMU releases (imuposer /
            dipimu / ncsa) are not in ``DEFAULT_TARGET_DATASETS``, so eval keeps
            everything-but-the-deny-list instead.
        split: ``val`` / ``test`` (only affects the train-only too-short filter,
            which is off for eval anyway).
        fps / imu_seq_max_len / motion_only / scene_only / dynamic_object: Passed
            to ``process_imu_data``.
        rewritten_texts: Optional ``sample id -> [caption]`` mapping from
            sidecars. Missing ids retain their original WDS captions.
        min_clip_length: Shortest clip (frames) to keep. ``None`` (default,
            historical) keeps only clips of at least ``cut_length`` frames. A
            smaller value keeps clips in ``[min_clip_length, cut_length)`` whole,
            so long windows (e.g. 480) still cover the complete test split.
        sample_id_regex: Optional regex applied to the complete sample id. Use
            it to select a nested subset inside a shared packed root.
        sample_ids: Optional exact sample-id allow-list. Use this for a fixed,
            reusable evaluation subset; order does not matter because shards
            retain their deterministic order.
        require_rewritten_text: Keep only ids present in ``rewritten_texts``.
            This prevents evaluation from falling back to raw WDS captions.
        imu_noise_cfg: ``training.imu_noise`` mapping; outside the train split
            only its fixed ``eval_lowpass_hz`` applies (synthetic sources).
        smooth_real_imu_acc: See :func:`_make_decode_process_fn`.
        contiguous_windows: Expand each selected packed sequence into adjacent,
            non-overlapping windows of ``cut_length``. This is intended for
            real-IMU benchmark sessions, which are packed as long recordings.
        contiguous_window_min_frames: Drop a final contiguous tail shorter than
            this many frames. A retained tail is passed through the usual
            tokenization path, which may trim a further 1--3 frames to the
            model's four-frame stride.
        allowed_keys: Optional WDS ``__key__`` allow-list from
            :func:`restrict_eval_shards`. Members outside it are dropped before
            ``.decode()``, i.e. never unpickled; the id-level filters above
            still run on what remains.

    Returns:
        A ``webdataset.WebLoader`` yielding single-element ``list[dict]`` batches.
    """
    if min_clip_length is not None and int(min_clip_length) > int(cut_length):
        raise ValueError(
            f"min_clip_length ({min_clip_length}) must not exceed cut_length ({cut_length})"
        )
    if contiguous_window_min_frames <= 0:
        raise ValueError(
            "contiguous_window_min_frames must be positive, got "
            f"{contiguous_window_min_frames}"
        )
    fixed_cut = SharedCut(cut_length)
    allowed_prefixes = set(target_datasets) if target_datasets is not None else None
    excluded_prefixes = set(DEFAULT_EXCLUDED_DATASETS) - (allowed_prefixes or set())
    allowed_sample_ids = None if sample_ids is None else set(sample_ids)
    if allowed_sample_ids is not None and not allowed_sample_ids:
        raise ValueError("sample_ids must contain at least one id when provided")
    decode_fn = _make_decode_process_fn(
        shared_cut=fixed_cut,
        split=split,
        random_cut=False,
        random_mask_text=False,
        shift=shift,
        filter_short_text=False,
        require_text=False,
        motion_only=motion_only,
        scene_only=scene_only,
        dynamic_object=dynamic_object,
        fps=fps,
        imu_seq_max_len=imu_seq_max_len,
        add_imu_noise=False,
        acc_scale=1.0,
        gyro_scale=1.0,
        allowed_prefixes=allowed_prefixes,
        excluded_prefixes=excluded_prefixes,
        rewritten_texts=rewritten_texts,
        min_clip_length=min_clip_length,
        sample_id_regex=sample_id_regex,
        allowed_sample_ids=allowed_sample_ids,
        require_rewritten_text=require_rewritten_text,
        imu_noise_cfg=imu_noise_cfg,
        smooth_real_imu_acc=smooth_real_imu_acc,
        real_heading_aug=real_heading_aug,
        short_window_global_supervision=short_window_global_supervision,
    )
    dataset = wds.WebDataset(shards, resampled=False, shardshuffle=False)
    if allowed_keys is not None:
        # Before .decode(): a dropped member costs a tar header read, not an
        # unpickle of its ~100 KB payload.
        dataset = dataset.select(lambda sample: sample["__key__"] in allowed_keys)
    dataset = dataset.decode()
    if contiguous_windows:
        def expand_contiguous_windows(samples):
            for sample in samples:
                payload = sample["sample.pkl"]
                try:
                    frame_count = len(payload["motion_data_smpl85"])
                except (KeyError, TypeError) as error:
                    raise ValueError(
                        "Contiguous-window evaluation requires motion_data_smpl85 "
                        f"for sample {payload.get('id', '<unknown>')!r}"
                    ) from error
                for start in range(0, frame_count, cut_length):
                    end = min(start + cut_length, frame_count)
                    if end - start < contiguous_window_min_frames:
                        continue
                    window_sample = dict(sample)
                    window_sample["__imu4d_eval_window_start"] = start
                    window_sample["__imu4d_eval_window_end"] = end
                    yield window_sample

        dataset = dataset.compose(expand_contiguous_windows)
    dataset = (
        dataset.map(decode_fn)
        .select(lambda x: x is not None)
        .batched(1, collation_fn=lambda batch: list(batch), partial=True)
    )
    return wds.WebLoader(dataset, batch_size=None, num_workers=num_workers)
