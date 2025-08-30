#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=100G
#SBATCH --job-name=srvae_5m_vs_6m
#SBATCH --cpus-per-task=2

# use srun to launch Jupyter server in order to reserve a port
echo "hello"
conda activate pymarl

echo "print SLURM variables, hostnames and GPUs"
env | grep SLURM | while read line; do echo "# $line"; done
hostname
nvidia-smi -L

cd ./pymarl2

python3 src/main.py --config=srvae_nqmix --env-config=sc2 with env_args.map_name=5m_vs_6m > $SLURM_JOB_ID/experiment_1.out 2> $SLURM_JOB_ID/experiment_1.err &
python3 src/main.py --config=srvae_nqmix --env-config=sc2 with env_args.map_name=5m_vs_6m > $SLURM_JOB_ID/experiment_2.out 2> $SLURM_JOB_ID/experiment_2.err &
python3 src/main.py --config=srvae_nqmix --env-config=sc2 with env_args.map_name=5m_vs_6m > $SLURM_JOB_ID/experiment_3.out 2> $SLURM_JOB_ID/experiment_3.err &