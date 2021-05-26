#!/bin/bash
#SBATCH -t 10:00:00
#SBATCH -p gpu_titanrtx_shared_course
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=32000M
#SBATCH --output=slurm_output%A.out

module purge
module load 2020
module load Python
module load Anaconda3/2020.02

source activate atcs2

srun python -u train_multitask.py run1_ag --finetuned_layers=-1 --num_workers=12 --batch_size=25 --lr=0.01 --train_sets=hp,dbpedia,ng --test_set=ag --max_epochs=10
