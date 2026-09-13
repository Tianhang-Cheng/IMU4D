import os
from pathlib import Path

# Repository-local resources.
dataset_process_root = 'dataset_process'

# The canonical tree is owned by the sibling IMU4D_dev repository.
code_root = Path(__file__).resolve().parents[2]
data_root = Path(
    os.environ.get('IMU4D_DATA_ROOT', code_root / 'IMU4D_dev' / 'data')
).expanduser()


def _data_path(env_name: str, relative_path: str) -> str:
    return os.environ.get(env_name, str(data_root / relative_path))

# smplx.SMPLX expects the directory that contains SMPLX_NEUTRAL.npz (same as `smplx_model_path` parent).
smplx_model_path = _data_path('IMU4D_SMPLX_PATH', 'models/smplx/SMPLX_NEUTRAL.npz')
smplx_model_dir = os.path.dirname(smplx_model_path)

# dataset contains 3D scene
humoto_root = _data_path('IMU4D_HUMOTO_ROOT', 'raw/humoto/v1/humoto_data')
humoto_objects_folder = _data_path(
    'IMU4D_HUMOTO_OBJECTS_ROOT', 'raw/humoto/v1/objects'
)
obj_name_path = f'{dataset_process_root}/obj_names_combine.txt'

# realworld or specific dataset
parahome_root = _data_path('IMU4D_PARAHOME_ROOT', 'processed/parahome/v1')
imuposer_root = _data_path('IMU4D_IMUPOSER_ROOT', 'processed/imuposer/v1')
dipimu_root = _data_path('IMU4D_DIPIMU_ROOT', 'processed/dipimu/v1')

# general imu dataset
imu_data_path = _data_path(
    'IMU4D_MOTION_DATA_ROOT', 'processed/motionmillion/eval_wds_v1'
)

# pretrained weights for showo
pretrained_showo_path = f'pretrained_weight/showo'
