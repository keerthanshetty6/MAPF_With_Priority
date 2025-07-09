import os

PBS_TEMPLATE = """#!/bin/bash
#SBATCH --output=logs/out_%j.out
#SBATCH --error=logs/err_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=5
#SBATCH --partition=long

set -e

module load miniconda
source /mnt/beegfs/apps/miniconda/etc/profile.d/conda.sh
conda activate cmapf-env

echo "Running job {job_id} on $(hostname)"
{command}
"""

commands_txt = "mapf_job_commands.txt"
pbs_dir = "jobs"
max_jobs = 3  # only generate first 3 jobs -> max_jobs = None

os.makedirs(pbs_dir, exist_ok=True)

with open(commands_txt, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

print(f"Found {len(lines)} commands. Generating up to {max_jobs} PBS files for testing.")

for idx, command in enumerate(lines if max_jobs is None else lines[:max_jobs], 1):
    job_id = f"{idx:04d}"  # zero-padded
    pbs_content = PBS_TEMPLATE.format(job_id=job_id, command=command)
    pbs_filename = os.path.join(pbs_dir, f"job_{job_id}.pbs")
    with open(pbs_filename, "w") as pbs_file:
        pbs_file.write(pbs_content)

print(f"Generated {min(len(lines), max_jobs)} PBS job files in '{pbs_dir}/'.")
