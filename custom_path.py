import os

dataset_process_root = '/home/tcheng12/code/imu-human-mllm/dataset_process'

smplx_model_path = f'{dataset_process_root}/SMPLX_NEUTRAL.npz'
# smplx.SMPLX expects the directory that contains SMPLX_NEUTRAL.npz (same as `smplx_model_path` parent).
smplx_model_dir = os.path.dirname(smplx_model_path)
humoto_root = f'{dataset_process_root}/humoto_data'
parahome_root = f'{dataset_process_root}/ParaHome'
imuposer_root = f'{dataset_process_root}/imuposer_dataset_processed'
dipimu_root = f'{dataset_process_root}/DIP_IMU_processed'
obj_name_path = f'{dataset_process_root}/obj_names_combine.txt'
pretrained_showo_path = f'pretrained_weight/showo'

# Object meshes for visualize/viz_motion.py (Humoto scene objects; set to your local download).
humoto_objects_folder = '/path/to/humoto_objects_0805'

imu_data_path = '/shared/perception/datasets/imu_data/final_data_per_sequence'
