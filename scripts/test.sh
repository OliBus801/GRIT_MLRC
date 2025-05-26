#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --account=aip-chgag196
#SBATCH --gpus-per-node=h100:4

module load python/3.10
module load cuda
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install torch==2.1.1 --no-index

python scripts/pytorch-test.py