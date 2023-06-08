#!/bin/sh
#SBATCH --job-name=mridc
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --nice=10

# conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /scratch/mkingma/envs/mridc1

# job
nice -n 10 python -m mridc.launch --config-path /data/projects/recon/data/private/neonatal/mkingma/code/launchfiles/ --config-name CIRIM-gaussian1D.yaml

# Wait is needed to make sure slurm doesn't quit the job when the lines with '&' immediately return
wait
