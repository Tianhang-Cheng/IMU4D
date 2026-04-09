import os

# TODO: replace to your path
dataset_process_root = 'dataset_process'

# smplx.SMPLX expects the directory that contains SMPLX_NEUTRAL.npz (same as `smplx_model_path` parent).
smplx_model_path = f'{dataset_process_root}/SMPLX_NEUTRAL.npz'
smplx_model_dir = os.path.dirname(smplx_model_path)

# dataset contains 3D scene
humoto_root = f'{dataset_process_root}/humoto_data'
humoto_objects_folder = '/path/to/humoto_objects_0805'
obj_name_path = f'{dataset_process_root}/obj_names_combine.txt'

# realworld or specific dataset
parahome_root = f'{dataset_process_root}/ParaHome'
imuposer_root = f'{dataset_process_root}/imuposer_dataset_processed'
dipimu_root = f'{dataset_process_root}/DIP_IMU_processed'

# general imu dataset
imu_data_path = '/shared/perception/datasets/imu_data/final_data_per_sequence'

# pretrained weights for showo
pretrained_showo_path = f'pretrained_weight/showo'
