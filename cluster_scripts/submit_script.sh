#!/bin/bash
#SBATCH --job-name=OHK_DMRG      # Job name
#SBATCH --output=logs/output_%j.log    # Output log file (%j will be replaced with the job ID)
#SBATCH --error=errors/error_%j.log      # Error log file (%j will be replaced with the job ID)
#SBATCH --ntasks=1                # Number of tasks (usually 1 for a single script)
#SBATCH --cpus-per-task=64         # Number of CPU cores per task
#SBATCH --time=12:00:00           # Time limit (hh:mm:ss)
#SBATCH --partition=IllinoisComputes      # Partition name
#SBATCH --account=tenkila2-ic

# Load required modules
module load python/3
export PYTHONPATH=/home/${USER}/mypython3:${PYTHONPATH}

# Run your Python script
python3 -u temp.py 

