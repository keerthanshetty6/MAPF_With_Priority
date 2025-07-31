import os
import sys

PBS_TEMPLATE = """#!/bin/bash
#SBATCH --output=logs/out_%j.out
#SBATCH --error=logs/err_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=5
#SBATCH --partition=long
#SBATCH --exclusive

set -e

module load miniconda
source /mnt/beegfs/apps/miniconda/etc/profile.d/conda.sh
conda activate cmapf-env

echo "SLURM JobID: $SLURM_JOB_ID"
echo "Conda binary: $(which conda)"
echo "Python binary: $(which python)"
echo "Running job {job_id} on $(hostname)"

{command}
"""

# Parse command line arguments
if len(sys.argv) > 1:
    commands_txt = sys.argv[1]
else:
    commands_txt = "mapf_jobs.txt"

if len(sys.argv) > 2:
    pbs_dir = sys.argv[2]
else:
    pbs_dir = "jobs"

max_jobs = None

os.makedirs(pbs_dir, exist_ok=True)

with open(commands_txt, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

print(f"Found {len(lines)} commands. Generating PBS files...")

for idx, command in enumerate(lines if max_jobs is None else lines[:max_jobs], 1):
    job_id = f"{idx:04d}"  # zero-padded
    pbs_content = PBS_TEMPLATE.format(job_id=job_id, command=command)
    pbs_filename = os.path.join(pbs_dir, f"job_{job_id}.pbs")
    with open(pbs_filename, "w") as pbs_file:
        pbs_file.write(pbs_content)

limit = max_jobs if max_jobs is not None else len(lines)
print(f"Found {len(lines)} commands. Generating up to {limit} PBS files.")
