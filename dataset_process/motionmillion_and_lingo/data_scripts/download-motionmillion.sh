# Install git-lfs without sudo
cd ~ && \
wget https://github.com/git-lfs/git-lfs/releases/download/v3.4.1/git-lfs-linux-amd64-v3.4.1.tar.gz && \
tar -xzf git-lfs-linux-amd64-v3.4.1.tar.gz && \
mkdir -p ~/bin && \
cp git-lfs-3.4.1/git-lfs ~/bin/ && \
chmod +x ~/bin/git-lfs && \
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc && \
source ~/.bashrc && \
rm -rf git-lfs-3.4.1 git-lfs-linux-amd64-v3.4.1.tar.gz


# Download motionmillion dataset
git lfs install
git config --global credential.helper store 
git clone https://huggingface.co/datasets/InternRobotics/MotionMillion
cd MotionMillion
git lfs pull


# Extract all tar.gz files in their respective directories
bash unzip-motionmillion.sh
tar -xzvf texts.tar.gz


# Download and process HumanML3D and BABEL dataset following MotionStreamer (https://github.com/zju3dv/MotionStreamer)
huggingface-cli download --repo-type dataset --resume-download lxxiao/272-dim-HumanML3D --local-dir ./humanml3d_272
cd ./humanml3d_272
unzip texts.zip
unzip motion_data.zip

python data_process/HumanML3D/preprocess_cut_humanml3d.py \
    --text_root ./humanml3d_272/texts \
    --motion_root ./humanml3d_272/motion_data \
    --output_text_root ./humanml3d_cutted/texts \
    --output_motion_root ./humanml3d_cutted/motion_data

mv ./humanml3d_cutted/motion_data ./motion_272rpr/MotionUnion/humanml

python data_process/HumanML3D/preprocess_mirror_humanml3d.py \
    --text_root ./humanml3d_cutted/texts \
    --motion_root ./motion_272rpr/MotionUnion/humanml \
    --output_text_root ./mirror_humanml3d_cutted/texts \
    --output_motion_root ./mirror_humanml3d_cutted/motion_data   # ING

mv ./mirror_humanml3d_cutted/motion_data ./motion_272rpr/Mirror_MotionUnion/humanml


# Note: BABEL only uses for motion tokenizer training, thus texts are not processed
huggingface-cli download --repo-type dataset --resume-download lxxiao/272-dim-BABEL --local-dir ./babel_272
cd ./babel_272
unzip texts.zip
unzip motion_data.zip
python data_process/BABEL/preprocess_mirror_babel.py \
    --motion_root ./babel_272/motion_data \
    --output_motion_root ./mirror_babel_272/motion_data

mkdir -p ./motion_272rpr/BABEL
mkdir -p ./motion_272rpr/Mirror_BABEL
mv ./babel_272/motion_data ./motion_272rpr/BABEL/new_joint_vecs_prefix
mv ./mirror_babel_272/motion_data ./motion_272rpr/Mirror_BABEL/new_joint_vecs_prefix


# Download and process AIST dataset from MotionHub-V2
# Hard to get AIST dataset as we need to download baidu pan first...orz
unzip aist.zip
python data_process/AIST/s1_extract_smpl85.py \
    --root_dir ./aist \
    --output_dir ./aist_smpl85
python data_process/AIST/s2_mirror_smpl85.py \
    --root_dir ./aist_smpl85 \
    --output_dir ./mirror_aist_smpl85
python data_process/AIST/s3_downsample_to_30fps.py \
    --root_dir ./aist_smpl85 \
    --output_dir ./aist_smpl85_30fps
python data_process/AIST/s3_downsample_to_30fps.py \
    --root_dir ./mirror_aist_smpl85 \
    --output_dir ./mirror_aist_smpl85_30fps

mv ./aist_smpl85_30fps/standard_smplx ./aist_smpl85_30fps/smpl_85
mv ./mirror_aist_smpl85_30fps/standard_smplx ./mirror_aist_smpl85_30fps/smpl_85

git clone https://github.com/Li-xingXiao/272-dim-Motion-Representation.git
pip install human-body-prior@git+https://github.com/nghorbani/human_body_prior@4c246d8a83ce16d3cff9c79dcf04d81fa440a6bc
# download SMPLX_NEUTRAL.npz from ...

python 272-dim-Motion-Representation/face_z_transform.py --filedir ./aist_smpl85_30fps
python 272-dim-Motion-Representation/infer_get_joints.py --filedir ./aist_smpl85_30fps
python 272-dim-Motion-Representation/representation_272.py --filedir ./aist_smpl85_30fps

python 272-dim-Motion-Representation/face_z_transform.py --filedir ./mirror_aist_smpl85_30fps
python 272-dim-Motion-Representation/infer_get_joints.py --filedir ./mirror_aist_smpl85_30fps
python 272-dim-Motion-Representation/representation_272.py --filedir ./mirror_aist_smpl85_30fps

mkdir -p ./motion_272rpr/MotionUnion/aist
mkdir -p ./motion_272rpr/Mirror_MotionUnion/aist
mv ./aist_smpl85_30fps/Representation_272 ./motion_272rpr/MotionLLAMA/aist/standard_smplx
mv ./mirror_aist_smpl85_30fps/Representation_272 ./motion_272rpr/Mirror_MotionLLAMA/aist/standard_smplx



# No need to convert to 272-dim, as AIST is already in 85-dim with Y-up 30 fps
# git clone https://github.com/Li-xingXiao/272-dim-Motion-Representation.git