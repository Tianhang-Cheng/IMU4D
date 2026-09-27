"""Rotation conversion and SO(3) distance utilities."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .common import as_float_array


def rotation_matrix(value: np.ndarray, representation: str) -> np.ndarray:
    """Convert ``matrix``, WXYZ ``quaternion``, ``axis_angle``, or ``6d`` to SO(3).

    The 6D convention is the first two matrix columns concatenated, matching
    ``utils.rotation.matrix_to_cont_6d`` in this repository.
    """

    value = as_float_array(value, "rotation")
    representation = representation.lower().replace("-", "_")
    if representation in {"matrix", "mat"}:
        if value.shape[-2:] != (3, 3):
            raise ValueError(f"matrix rotations must end in [3,3], got {value.shape}")
        return value
    if representation in {"quaternion", "quat", "wxyz"}:
        if value.shape[-1] != 4:
            raise ValueError(f"WXYZ quaternions must end in [4], got {value.shape}")
        norm = np.linalg.norm(value, axis=-1, keepdims=True)
        if np.any(norm < 1e-12):
            raise ValueError("Quaternion norm must be non-zero")
        wxyz = value / norm
        xyzw = wxyz[..., [1, 2, 3, 0]]
        return Rotation.from_quat(xyzw.reshape(-1, 4)).as_matrix().reshape(
            value.shape[:-1] + (3, 3)
        )
    if representation in {"axis_angle", "aa", "rotvec"}:
        if value.shape[-1] != 3:
            raise ValueError(f"axis-angle rotations must end in [3], got {value.shape}")
        return Rotation.from_rotvec(value.reshape(-1, 3)).as_matrix().reshape(
            value.shape[:-1] + (3, 3)
        )
    if representation in {"6d", "continuous_6d", "cont_6d"}:
        if value.shape[-1] != 6:
            raise ValueError(f"6D rotations must end in [6], got {value.shape}")
        first = value[..., :3]
        second = value[..., 3:]
        first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
        if np.any(first_norm < 1e-12):
            raise ValueError("The first 6D rotation axis has zero norm")
        x_axis = first / first_norm
        y_axis = second - np.sum(second * x_axis, axis=-1, keepdims=True) * x_axis
        y_norm = np.linalg.norm(y_axis, axis=-1, keepdims=True)
        if np.any(y_norm < 1e-12):
            raise ValueError("The two 6D rotation axes are collinear")
        y_axis /= y_norm
        z_axis = np.cross(x_axis, y_axis)
        return np.stack([x_axis, y_axis, z_axis], axis=-1)
    raise ValueError(f"Unsupported rotation representation: {representation}")


def geodesic_distance_deg(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    prediction = as_float_array(prediction, "prediction rotations")
    target = as_float_array(target, "target rotations")
    if prediction.shape != target.shape or prediction.shape[-2:] != (3, 3):
        raise ValueError(
            f"Expected equal [...,3,3] rotation arrays, got {prediction.shape} and {target.shape}"
        )
    relative = prediction @ np.swapaxes(target, -1, -2)
    cosine = (np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def transform_points(
    points: np.ndarray, rotations: np.ndarray, translations: np.ndarray
) -> np.ndarray:
    points = as_float_array(points, "mesh points")
    rotations = as_float_array(rotations, "rotations")
    translations = as_float_array(translations, "translations")
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"mesh points must have shape [P,3], got {points.shape}")
    return np.einsum("...ij,pj->...pi", rotations, points) + translations[..., None, :]
