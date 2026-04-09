
datasets=(
    'BABEL' 
    'Mirror_BABEL' 
    'PhantomDanceDatav1.1' 
    'Mirror_PhantomDanceDatav1.1' 
    'MotionGV' 
    'MotionLLAMA' 
    'MotionUnion' 
    'Mirror_MotionGV' 
    'Mirror_MotionLLAMA' 
    'Mirror_MotionUnion',
    'LINGO'
)

# Loop through each dataset and run the Python script
for dataset in "${datasets[@]}"; do
    python get_imu_6dof_traj.py --dataset_name "$dataset"
done
