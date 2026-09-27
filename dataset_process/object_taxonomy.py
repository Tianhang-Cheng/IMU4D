"""Whole-object taxonomy and backward-compatible annotation normalization."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy


# Categories are whole-object semantic classes. Source-specific subtype and
# geometry information remains in source_category and asset_id.
CATEGORY_ALIASES = {
    # Cross-dataset spelling aliases.
    "basket_90": "basket",
    "clothrack": "clothes_rack",
    "stepstool": "step_stool",
    "trashbin": "trash_can",
    "ukelele": "ukulele",
    # High-confidence subtype normalization.
    "woven_basket": "basket",
    "bathroomsink": "sink",
    "dining_chair": "chair",
    "low_chair": "chair",
    "working_chair": "chair",
    "organizer_medium": "organizer",
    "organizer_small": "organizer",
    "kitchen_counter_small": "kitchen_counter",
    # Articulated parts are never semantic object categories.
    "laptop_bottom": "laptop",
    "laptop_top": "laptop",
    "soap_dispenser_body": "soap_dispenser",
    "soap_dispenser_top": "soap_dispenser",
    "vacuum_flask_body": "vacuum_flask",
    "vacuum_flask_cap": "vacuum_flask",
    "mop_bottom": "mop",
    "vacuum_bottom": "vacuum",
}


# Ordered root preference for families that legacy records represented as
# multiple objects. Only the selected root track survives in v1 whole-object
# training; no part trajectory or part token is exposed to the model.
PART_FAMILY_ROOT_PRIORITY = {
    "laptop": ("laptop", "laptop_bottom", "laptop_top"),
    "soap_dispenser": (
        "soap_dispenser",
        "soap_dispenser_body",
        "soap_dispenser_top",
    ),
    "vacuum_flask": (
        "vacuum_flask",
        "vacuum_flask_body",
        "vacuum_flask_cap",
    ),
    "mop": ("mop", "mop_bottom"),
    "vacuum": ("vacuum", "vacuum_bottom"),
}

PART_TO_PARENT = {
    member: parent
    for parent, members in PART_FAMILY_ROOT_PRIORITY.items()
    for member in members
    if member != parent
}


def source_category(name: str) -> str:
    """Strip an instance suffix while retaining the source taxonomy."""

    return str(name).strip().partition("__")[0]


def canonical_category(name: str) -> str:
    """Return the stable whole-object semantic category for a source name."""

    category = source_category(name)
    seen: set[str] = set()
    while category in CATEGORY_ALIASES:
        if category in seen:
            raise ValueError(f"Cyclic object category alias at {category!r}")
        seen.add(category)
        category = CATEGORY_ALIASES[category]
    return category


def is_part_category(name: str) -> bool:
    return source_category(name) in PART_TO_PARENT


# The synthetic ground plane that the loader adds for supported sources without a
# literal floor entry (dataset_process/wds_pipeline/wds_loader.py
# needs_synthetic_ground) is a per-sample CONSTANT by construction: the object
# crop repeats one pose over the whole clip and process_imu_data zeroes its X/Z.
# It therefore carries no per-frame object-track information and is excluded
# from the dynamic-track / motion-state supervision; its static pose, category
# token and identity remain supervised. Its pose/anchor is supervised only when
# the floor is known or its per-clip estimate passes the contact checks.
STATIC_GROUND_CATEGORIES = frozenset({"ground"})


def is_static_ground(name: str) -> bool:
    """Whether an object name denotes the constant ground plane."""

    return canonical_category(name) in STATIC_GROUND_CATEGORIES


def unique_instance_key(category: str, existing: Mapping[str, object]) -> str:
    """Return category, category__2, ... without collapsing same-class instances."""

    if category not in existing:
        return category
    suffix = 2
    while f"{category}__{suffix}" in existing:
        suffix += 1
    return f"{category}__{suffix}"


def canonical_asset_id(asset_id: str | None) -> str | None:
    """Map legacy part asset IDs to their whole-object retrieval target."""

    if asset_id is None:
        return None
    value = str(asset_id)
    legacy_aliases = {
        "humoto/laptop_bottom": "humoto/laptop",
        "humoto/laptop_top": "humoto/laptop",
        "humoto/soap_dispenser_body": "humoto/soap_dispenser",
        "humoto/soap_dispenser_top": "humoto/soap_dispenser",
        "humoto/vacuum_flask_body": "humoto/vacuum_flask",
        "humoto/vacuum_flask_cap": "humoto/vacuum_flask",
        "omomo/mop:top": "omomo/mop",
        "omomo/mop:bottom": "omomo/mop",
        "omomo/vacuum:top": "omomo/vacuum",
        "omomo/vacuum:bottom": "omomo/vacuum",
    }
    return legacy_aliases.get(value, value)


def normalize_object_annotations(
    objects: Mapping[str, object],
    metadata: Mapping[str, Mapping[str, object]] | None = None,
    valid_masks: Mapping[str, object] | None = None,
    motion_masks: Mapping[str, object] | None = None,
    *,
    source: str | None = None,
) -> tuple[dict[str, object], dict[str, dict], dict[str, object], dict[str, object]]:
    """Normalize old annotations to one track and one token per whole object.

    Same-category independent instances remain separate through ``__N`` keys.
    Only known articulated part families are collapsed, using the documented
    root priority above. The function is non-mutating and can therefore be
    applied safely to both legacy WDS payloads and new converter output.
    """

    metadata = metadata or {}
    valid_masks = valid_masks or {}
    motion_masks = motion_masks or {}
    ordered_keys = list(objects)
    key_order = {key: index for index, key in enumerate(ordered_keys)}
    consumed: set[str] = set()
    groups: list[tuple[int, str, list[str]]] = []

    for parent, priority in PART_FAMILY_ROOT_PRIORITY.items():
        family_keys = [
            key for key in ordered_keys if source_category(key) in set(priority)
        ]
        # Parent-category objects alone may be independent same-class
        # instances. Collapse only when a legacy part label is actually
        # present in the family.
        if not family_keys or not any(
            source_category(key) in PART_TO_PARENT for key in family_keys
        ):
            continue
        root_key = next(
            key
            for preferred in priority
            for key in family_keys
            if source_category(key) == preferred
        )
        groups.append((min(key_order[key] for key in family_keys), root_key, family_keys))
        consumed.update(family_keys)

    groups.extend(
        (key_order[key], key, [key]) for key in ordered_keys if key not in consumed
    )
    groups.sort(key=lambda item: item[0])

    normalized_objects: dict[str, object] = {}
    normalized_metadata: dict[str, dict] = {}
    normalized_valid: dict[str, object] = {}
    normalized_motion: dict[str, object] = {}

    for _, root_key, source_keys in groups:
        root_source = source_category(root_key)
        category = canonical_category(root_source)
        instance_key = unique_instance_key(category, normalized_objects)
        normalized_objects[instance_key] = objects[root_key]

        root_metadata = deepcopy(dict(metadata.get(root_key, {})))
        original_source = str(root_metadata.get("source_category", root_source))
        root_metadata["category"] = category
        root_metadata["source_category"] = original_source
        if len(source_keys) > 1:
            root_metadata["merged_source_categories"] = [
                source_category(key) for key in source_keys
            ]
            root_metadata["whole_object_root_source"] = root_source
            root_metadata.pop("mesh_path", None)

        asset_id = canonical_asset_id(root_metadata.get("asset_id"))
        if source == "humoto" and len(source_keys) > 1:
            asset_id = f"humoto/{category}"
        elif source == "omomo":
            asset_id = f"omomo/{original_source}"
            root_metadata["part"] = "whole"
            if original_source in {"mop", "vacuum"}:
                root_metadata["mesh_file"] = (
                    f"{original_source}_cleaned_simplified.obj"
                )
        if asset_id is not None:
            root_metadata["asset_id"] = asset_id
        normalized_metadata[instance_key] = root_metadata

        if root_key in valid_masks:
            normalized_valid[instance_key] = valid_masks[root_key]
        if root_key in motion_masks:
            normalized_motion[instance_key] = motion_masks[root_key]

    return (
        normalized_objects,
        normalized_metadata,
        normalized_valid,
        normalized_motion,
    )
