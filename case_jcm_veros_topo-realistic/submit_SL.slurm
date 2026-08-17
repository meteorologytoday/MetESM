#!/bin/bash
#
#SBATCH --job-name=JEM-SL-real-topo
#SBATCH --nodes 1
#SBATCH --tasks 1
#SBATCH --partition=gpu
#SBATCH --qos=gpumedium
#SBATCH --gres=gpu:1
#SBATCH --output=sbatch.64.%j.out
#SBATCH --mail-type=ALL,TIME_LIMIT
#SBATCH --time=72:00:00

nvidia-smi

echo "pwd: $(pwd)"
eval "$( /home/tienyiao/miniconda3/bin/conda shell.bash hook 2> /dev/null)"
conda activate jax

source /home/tienyiao/.bashrc_jcm_v2
export PYTHONPATH=/home/tienyiao/projects_local/project_jax-esm/jem_repo/fix-issue-99:$PYTHONPATH

# Semi-Lagrangian test setup: prepend the SL jax-gcm (dev branch, PR #625) and
# the shoyer/dinosaur@semi-lagrangian fork ahead of everything else on
# PYTHONPATH, so they shadow the pip-installed main-branch jcm / PyPI
# dinosaur==1.3.6 in the `jax` conda env. Comment this out to go back to the
# normal (Eulerian) setup.
export PYTHONPATH=/home/tienyiao/projects_local/project_jax-esm/jcm_collections/semi-lagrangian/jax-gcm:/home/tienyiao/projects_local/project_jax-esm/jcm_collections/semi-lagrangian/dinosaur:$PYTHONPATH

export PYTHONHASHSEED=0
#export XLA_FLAGS="--xla_gpu_deterministic_ops=true"

echo "PYTHONPATH=$PYTHONPATH"

JOB_ID=${SLURM_JOBID:-NO_JOB_ID}

time python3 main_forward.py \
    --total-simulation-days $(( 365 * 500 ))        \
    --simulation-interval-days 365                  \
    --truncation-number 31                          \
    --simulation-name real_topo_SL                  \
    --jcm-timestep-min 30                           \
    --veros-timestep-min 60                         \
    --max-rerun-attempts 0                          \
    --grid-folder data                              \
    --explode-log explode.${JOB_ID}.log

