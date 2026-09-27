"""Shared geometry-identity catalog for HUMOTO, HiPHI, and OMOMO.

Categories are semantic labels shared across datasets.  Asset identities are
source-qualified CAD identities: exported copies of the same CAD asset share
one identity while remaining resolvable through their original mesh names.
The catalog is deliberately JSON-backed so preprocessing, training, inference,
and visualization all use the same stable asset indices.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from dataset_process.omomo.omomo_io import CANONICAL_OBJECT_NAMES, parse_motion_id
from dataset_process.object_taxonomy import (
    canonical_asset_id,
    canonical_category,
    is_part_category,
)


SCHEMA_VERSION = 2

# Legacy processed data can mention these semantic classes even though the
# current HUMOTO/HiPHI/OMOMO geometry releases contain no corresponding CAD
# asset.  Keeping the vocabulary in the catalog removes the old standalone
# text-file dependency without making those samples unreadable.
EXTRA_SEMANTIC_CATEGORIES = frozenset({
    "bookshelf", "computer", "desk", "dishwasher", "faucet", "fridge",
    "microwave", "nightstand", "room", "sofa", "stove", "toilet",
})

HIPHI_MESH_PREFIX_TO_CATEGORY = {
    "Ball": "ball",
    "Bench": "bench",
    "Bottle": "bottle",
    "Box": "box",
    "Bucket": "bucket",
    "Chair": "chair",
    "Clothrack": "clothes_rack",
    "Mop": "mop",
    # The public release uses this misspelling in its mesh filename.
    "ScoccerBall": "soccerball",
    "StepStool": "step_stool",
    "Table": "table",
    "Trashbin": "trash_can",
}

# HiPHI sometimes includes multiple scene-exported copies of the same CAD
# mesh.  These names differ only by the source transform / floating-point OBJ
# serialization and therefore must share one retrieval target.
ASSET_ID_ALIASES = {
    "hiphi/Ball_A_2": "hiphi/Ball_A_1",
    "hiphi/Ball_A_3": "hiphi/Ball_A_1",
    "hiphi/Box_I_2": "hiphi/Box_I_1",
}


def _obj_extent(path: Path, unit_scale: float = 1.0) -> list[float]:
    vertices: list[list[float]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("v "):
                fields = line.split()
                vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
    if not vertices:
        raise ValueError(f"OBJ contains no vertices: {path}")
    array = np.asarray(vertices, dtype=np.float64) * unit_scale
    return (array.max(axis=0) - array.min(axis=0)).tolist()


def _asset(
    asset_id: str,
    category: str,
    dataset: str,
    mesh_path: str | None,
    extent_m: Sequence[float] | None,
    **metadata: object,
) -> dict[str, object]:
    return {
        "asset_id": asset_id,
        "category": canonical_category(category),
        # Explicit source label for readers of the JSON.  `dataset` remains
        # the machine-facing field retained for backwards compatibility.
        "source": dataset,
        "dataset": dataset,
        "mesh_path": mesh_path,
        "extent_m": None if extent_m is None else [float(x) for x in extent_m],
        **metadata,
    }


def _humoto_assets(root: Path) -> list[dict[str, object]]:
    mesh_root = root / "humoto_objects_0805"
    assets = [
        _asset(
            f"humoto/{path.stem}",
            path.stem,
            "humoto",
            str(path.relative_to(root)),
            _obj_extent(path),
        )
        for path in sorted(mesh_root.glob("*/*.obj"))
        if not is_part_category(path.stem)
    ]
    # Ground is present in every converted HUMOTO sample but has no CAD mesh.
    assets.append(_asset("humoto/ground", "ground", "humoto", None, None))
    return assets


def _hiphi_assets(root: Path) -> list[dict[str, object]]:
    assets: list[dict[str, object]] = []
    for path in sorted((root / "object_meshes").glob("*.obj")):
        if path.stem.endswith("__mirror"):
            continue
        prefix = path.stem.split("_", 1)[0]
        try:
            category = HIPHI_MESH_PREFIX_TO_CATEGORY[prefix]
        except KeyError as error:
            raise ValueError(f"Unknown HiPHI mesh prefix {prefix!r}: {path}") from error
        assets.append(
            _asset(
                f"hiphi/{path.stem}",
                category,
                "hiphi",
                str(path.relative_to(root)),
                _obj_extent(path, unit_scale=0.01),
            )
        )
    return assets


def _omomo_reference_scales(root: Path) -> dict[tuple[str, str], float]:
    """Compute one representative release scale per OMOMO mesh part."""

    import joblib

    scales: dict[tuple[str, str], list[float]] = defaultdict(list)
    for split in ("train", "test"):
        path = root / f"{split}_diffusion_manip_seq_joints24.p"
        if not path.is_file():
            continue
        sequences = joblib.load(path)
        for sequence in sequences.values():
            _, source_name = parse_motion_id(str(sequence["seq_name"]))
            for part, prefix in (("top", "obj"), ("bottom", "obj_bottom")):
                key = f"{prefix}_scale"
                if key not in sequence:
                    continue
                value = np.asarray(sequence[key], dtype=np.float64)
                value = value[np.isfinite(value)]
                if value.size:
                    # Weight each sequence equally rather than long clips more.
                    scales[(source_name, part)].append(float(np.median(value)))
    return {key: float(np.median(values)) for key, values in scales.items()}


def _omomo_assets(root: Path) -> list[dict[str, object]]:
    mesh_root = root / "captured_objects"
    reference_scales = _omomo_reference_scales(root)
    assets: list[dict[str, object]] = []
    for source_name, category in sorted(CANONICAL_OBJECT_NAMES.items()):
        mesh_path = mesh_root / f"{source_name}_cleaned_simplified.obj"
        if not mesh_path.is_file():
            raise FileNotFoundError(mesh_path)
        reference_scale = reference_scales.get((source_name, "top"))
        raw_extent = np.asarray(_obj_extent(mesh_path), dtype=np.float64)
        extent = None if reference_scale is None else raw_extent * reference_scale
        assets.append(
            _asset(
                f"omomo/{source_name}",
                category,
                "omomo",
                str(mesh_path.relative_to(root)),
                extent,
                source_category=source_name,
                part="whole",
                reference_scale=reference_scale,
            )
        )
    return assets


def build_catalog(
    humoto_root: Path,
    hiphi_root: Path,
    omomo_root: Path,
) -> dict[str, object]:
    """Build a deterministic global catalog from the three releases."""

    assets = [
        *_humoto_assets(humoto_root),
        *_hiphi_assets(hiphi_root),
        *_omomo_assets(omomo_root),
    ]
    assets.sort(key=lambda item: str(item["asset_id"]))
    aliases_by_canonical: dict[str, list[str]] = defaultdict(list)
    canonical_assets: list[dict[str, object]] = []
    for asset in assets:
        asset_id = str(asset["asset_id"])
        canonical_id = ASSET_ID_ALIASES.get(asset_id, asset_id)
        aliases_by_canonical[canonical_id].append(asset_id)
        if canonical_id == asset_id:
            canonical_assets.append(asset)
    missing_canonicals = sorted(set(aliases_by_canonical) - {
        str(asset["asset_id"]) for asset in canonical_assets
    })
    if missing_canonicals:
        raise ValueError(f"Identity aliases lack canonical assets: {missing_canonicals}")
    assets = canonical_assets
    seen: set[str] = set()
    for index, asset in enumerate(assets):
        asset_id = str(asset["asset_id"])
        if asset_id in seen:
            raise ValueError(f"Duplicate asset identity: {asset_id}")
        seen.add(asset_id)
        aliases = aliases_by_canonical[asset_id]
        if len(aliases) > 1:
            asset["asset_aliases"] = aliases
        asset["asset_index"] = index

    category_counts: dict[str, int] = defaultdict(int)
    for asset in assets:
        category_counts[str(asset["category"])] += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "identity_embedding": "learned_during_training",
        "categories": dict(sorted(category_counts.items())),
        "category_vocabulary": sorted(
            set(category_counts) | set(EXTRA_SEMANTIC_CATEGORIES)
        ),
        "assets": assets,
    }


@dataclass(frozen=True)
class IdentityCatalog:
    assets: tuple[Mapping[str, object], ...]
    categories: tuple[str, ...]

    def __post_init__(self) -> None:
        ids = [str(asset["asset_id"]) for asset in self.assets]
        if len(ids) != len(set(ids)):
            raise ValueError("Identity catalog contains duplicate asset_id values")
        indices = [int(asset["asset_index"]) for asset in self.assets]
        if indices != list(range(len(self.assets))):
            raise ValueError("Identity catalog asset_index values must be contiguous")

    @classmethod
    def load(cls, path: Path | str) -> "IdentityCatalog":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(f"Unsupported identity catalog schema: {path}")
        asset_categories = {str(asset["category"]) for asset in payload["assets"]}
        categories = tuple(payload.get("category_vocabulary", sorted(asset_categories)))
        if not asset_categories.issubset(categories):
            missing = sorted(asset_categories - set(categories))
            raise ValueError(f"Catalog vocabulary excludes asset categories: {missing}")
        return cls(tuple(payload["assets"]), categories)

    @property
    def asset_id_to_index(self) -> dict[str, int]:
        indices: dict[str, int] = {}
        for asset in self.assets:
            index = int(asset["asset_index"])
            for asset_id in (str(asset["asset_id"]), *asset.get("asset_aliases", ())):
                previous = indices.setdefault(str(asset_id), index)
                if previous != index:
                    raise ValueError(f"Asset alias resolves ambiguously: {asset_id}")
        return indices

    def candidates(self, category: str) -> list[int]:
        category = canonical_category(category)
        return [
            int(asset["asset_index"])
            for asset in self.assets
            if asset["category"] == category
        ]

    def resolve_asset_id(
        self,
        object_name: str,
        metadata: Mapping[str, object] | None,
    ) -> str | None:
        """Resolve the three processed metadata schemas to one global ID."""

        metadata = metadata or {}
        if metadata.get("asset_id"):
            asset_id = canonical_asset_id(str(metadata["asset_id"]))
        elif metadata.get("mesh_id"):
            mesh_id = str(metadata["mesh_id"]).removesuffix("__mirror")
            asset_id = f"hiphi/{mesh_id}"
        elif metadata.get("source_category") and metadata.get("mesh_file"):
            source = str(metadata["source_category"])
            asset_id = f"omomo/{source}"
        else:
            return None
        if asset_id not in self.asset_id_to_index:
            return None
        return str(self.assets[self.asset_id_to_index[asset_id]]["asset_id"])

    def resolve_index(
        self,
        object_name: str,
        metadata: Mapping[str, object] | None,
    ) -> int:
        asset_id = self.resolve_asset_id(object_name, metadata)
        return -100 if asset_id is None else self.asset_id_to_index[asset_id]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--humoto-root", type=Path, default=Path("data/raw/humoto/v1")
    )
    parser.add_argument(
        "--hiphi-root", type=Path, default=Path("data/raw/hiphi/release_v1")
    )
    parser.add_argument(
        "--omomo-root", type=Path, default=Path("data/raw/omomo/release_v1/data")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset_process/object_identity_catalog.json"),
    )
    args = parser.parse_args()
    payload = build_catalog(args.humoto_root, args.hiphi_root, args.omomo_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(payload['assets'])} identities across "
        f"{len(payload['categories'])} categories to {args.output}"
    )


if __name__ == "__main__":
    main()
