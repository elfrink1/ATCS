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
#module load Anaconda3/2020.02

source activate atcs2

srun python -u train_multitask.py run1_lr0.0001 \
                                --finetuned_layers=-1 \
                                --num_workers=3 \
                                --batch_size=10 \
                                --lr=0.0001 \
                                --train_sets=hp,dbpedia,ng \
                                --test_set=bbc \
                                --max_epochs=40 \
                                --sample 15000
