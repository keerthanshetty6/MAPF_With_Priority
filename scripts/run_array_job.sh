#!/bin/bash
#SBATCH --output=logs/out_%A_%a.out
#SBATCH --error=logs/err_%A_%a.err
#SBATCH --time=02:00:00             # walltime
#SBATCH --cpus-per-task=5           # number of processor cores (i.e. tasks)
#SBATCH --partition=kr
#SBATCH --array=1-1296 # remove 

# Good Idea to stop operation on first error.
set -e

# Load environment modules for your application here.
source ~/.bashrc


module load miniconda
conda activate cmapf-env

echo "Running job ${SLURM_ARRAY_TASK_ID} on $(hostname)"

CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" mapf_job_commands.txt) # gets line equivalent to SLURM_ARRAY_TASK_ID from the file
echo "Executing: $CMD"
eval "$CMD" # Execute the command
