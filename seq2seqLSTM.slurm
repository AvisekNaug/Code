#!/bin/bash

#SBATCH --mail-user=avisek.naug@vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1 #no of cpu cores per task
#SBATCH --mem=64G #ram per node
#SBATCH --mem-per-cpu=64G #ram per cpu core #--time=2-00:00:00
#SBTACH --output=job_output1.out
#SBATCH --job-name=seq2seqLSTM
#SBATCH --time=8:00:00

source /home/nauga/anaconda3/bin/activate SmartBuildings

srun -N 1 -n 1 python stateful.py 5 &
srun -N 1 -n 1 python stateful.py 30 &
wait
