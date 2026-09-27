import os
from pathlib import Path

# Repository-local resources.
dataset_process_root = 'dataset_process'

# All datasets and licensed models use the repository's canonical data tree.
# IMU4D_DATA_ROOT remains available for an explicit server-local override.
repository_root = Path(__file__).resolve().parents[1]
data_root = Path(
    os.environ.get('IMU4D_DATA_ROOT', repository_root / 'data')
).expanduser()


def _data_path(env_name: str, relative_path: str) -> str:
    """Resolve an optional per-dataset override against the canonical root."""
    return os.environ.get(env_name, str(data_root / relative_path))

# smplx.SMPLX expects the directory that contains SMPLX_NEUTRAL.npz (same as `smplx_model_path` parent).
smplx_model_path = _data_path('IMU4D_SMPLX_PATH', 'models/smplx/SMPLX_NEUTRAL.npz')
smplx_model_dir = os.path.dirname(smplx_model_path)

# dataset contains 3D scene
hiphi_root = _data_path('IMU4D_HIPHI_ROOT', 'raw/hiphi/release_v1')
omomo_root = _data_path('IMU4D_OMOMO_ROOT', 'raw/omomo/release_v1/data')
humoto_root = _data_path('IMU4D_HUMOTO_ROOT', 'raw/humoto/v1/humoto_data')
humoto_objects_folder = _data_path(
    'IMU4D_HUMOTO_OBJECTS_ROOT', 'raw/humoto/v1/objects'
)

# realworld or specific dataset
parahome_root = _data_path('IMU4D_PARAHOME_ROOT', 'processed/parahome/v1')
# Legacy per-sequence pickles (P{1..10}/*.pkl, s_{01..10}/*.pkl) read by the
# map-style IMUDataset only; training/eval stream processed/<dataset>/v1/wds.
imuposer_root = _data_path('IMU4D_IMUPOSER_ROOT', 'raw/imuposer/legacy_v1')
dipimu_root = _data_path('IMU4D_DIPIMU_ROOT', 'raw/dipimu/legacy_v1')

# general imu dataset
imu_data_path = _data_path(
    'IMU4D_MOTION_DATA_ROOT', 'processed/motionmillion/v1'
)

# pretrained weights for showo
pretrained_showo_path = f'pretrained_weight/showo'
