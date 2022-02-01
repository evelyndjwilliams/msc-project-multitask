#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#  job name: -N
#$ -N test_fp_training
# This is where the log files will be stored, in this case, the current directory
#$ -cwd
# Running time
#$ -l h_rt=48:00:00
#  (working) memory limit, the amount of memory you need may vary given your code and the amount of data
#$ -l h_vmem=63G
# GPU environment
#$ -pe gpu 1

# Load Anaconda environment and modules
. /etc/profile.d/modules.sh
module load anaconda
source activate fp_venv
module load cuda/11.0.2
cd /exports/eddie/scratch/s1935763/msc-project-voice-conversion/CNNEMA

# Your python commands below...
python train.py