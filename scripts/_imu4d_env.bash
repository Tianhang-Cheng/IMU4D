# Shared environment bootstrap for scripts in this directory.

_imu4d_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_imu4d_repo_root="$(cd -- "$_imu4d_script_dir/.." && pwd)"
_imu4d_conda_env="${IMU4D_CONDA_ENV:-imu4d}"

# Logging stays local unless the user explicitly enables W&B.
export WANDB_MODE="${WANDB_MODE:-disabled}"

# Activate unless the shell already resolves python from the target env. Checking
# only CONDA_DEFAULT_ENV is not enough: a tmux server can inherit a stale
# CONDA_DEFAULT_ENV while PATH still points at another env's python.
_imu4d_python="$(command -v python || true)"
if [[ "${CONDA_DEFAULT_ENV:-}" != "$_imu4d_conda_env" || "$_imu4d_python" != */envs/"$_imu4d_conda_env"/bin/python ]]; then
    _imu4d_conda_exe="${CONDA_EXE:-$(command -v conda || true)}"
    [[ -n "$_imu4d_conda_exe" ]] || {
        echo "Conda was not found; cannot activate '$_imu4d_conda_env'." >&2
        return 1
    }
    source "$("$_imu4d_conda_exe" info --base)/etc/profile.d/conda.sh"
    conda activate "$_imu4d_conda_env"
fi

# Per-batch crop buckets change tensor shapes every step; expandable segments
# stop the CUDA caching allocator from fragmenting / over-reserving memory.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$_imu4d_repo_root"

unset _imu4d_conda_env _imu4d_conda_exe _imu4d_repo_root _imu4d_script_dir _imu4d_python
