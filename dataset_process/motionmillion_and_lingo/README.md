## MotionMillion & lingo dataset processing

Remember to change the `root_dir` and `out_dir` to the desired path.

Download the processed dataset and file splits from [here](https://drive.google.com/drive/folders/1-Lujn-DR4OJV2WdyCnT6BzvMxfEtxT3Q?usp=sharing).

### Dataset format
```bash
motionmillion_smpl85/
    - LINGO/
        - imu_traj.npy
        - motion_smpl85.npy
        - start_end.pkl
        - texts.pkl
    - MotionLLAMA/
        - aist/
        - finedance/
        - fit3d/
        ...
    - ...
motionmillion_splits/
    - t2m_train.txt
    - tokenizer_train.txt
    - ...
```
#### Details of each file under a specific dataset folder
- `imu_traj.npy`: 6 IMU 6-dof trajectories array of size (N, 6, 6), with the last dimension to be (axis-angle, position)
- `motion_smpl85.npy`: motions array of size (N, 85) with orient (3), pose (63), pose_wrist (6), transl (3), body_betas (10), where pose_wrist and body_betas are currently unused.
- `start_end.pkl`: a dictionary with file path as keys and (start, end) index as values. For example, use `motion_data = motion_smpl85[start:end, :]` to get a specific motion sequence.
- `texts.pkl`: a dictionary with file path as keys and list of descriptions as values. **Note that some datasets may not contain this file.**
- `t2m_xxx.txt`: train/val/test split of text-to-motion dataset.
- `tokenizer_xxx.txt`: train/val/test split of motion dataset (as some datasets may not contain text descriptions).


### MotionMillion dataset
Follow the steps listed in `data_scripts/download-motionmillion.sh` to:
1. Download available motionmillion dataset
2. Download other datasets (e.g., BABEL, HumanML3D, AIST) and process to 272-dim format
3. Process all available data to 85-dim format using `get_smpl85.py`
4. Run `get_texts.py` to aggregate all text descriptions information.
5. Run `get_smpl141.py` to get the compact 141-dim format from full 272-dim format.

### Lingo dataset
Follow the steps below:
1. Download `dataset.zip` from [this link](https://drive.google.com/file/d/1RadpLt-woPvsIGk9yDW7yX7br3sRO-zi/view).
2. Use `convert_lingo_dataset.py` to (1) align poses to the first frame, (2) convert to 85-dim format & 141-dim format, and (3) aggergate text descriptions.

### Simulate IMU 6-DoF trajectory
Use `bash data_scripts/get_imu_traj.sh` to simulate IMU trajectories for all datasets. Note that the required SMPL-X model could be downloaded from [here](https://drive.google.com/file/d/1ryYmi-ajAj5FcaT9pBvcJQZX2s9IzqMT/view?usp=sharing). SMPL-85 motions are used for trajectory simulation.

### Get train/val/test file splits
Run `get_splits.py` to make sure the newly added dataset (i.e., LINGO) appears in both text-to-motion and motion tokenizer motion splits.

### Filter motion dataset
Run `filter_data.py` to remove motion dataset which contains motions with pelvis jittering or unrealistic joint angles.

### Compute mean & std
Run `calculate_mean_std.py` to get the mean and std of the motion data. We average std values for each category following HumanML3D. Support smpl85, smpl141, and smpl272 format.

### Misc
Install `human_body_prior` using this command:
```bash
pip install human-body-prior@git+https://github.com/nghorbani/human_body_prior@4c246d8a83ce16d3cff9c79dcf04d81fa440a6bc
```