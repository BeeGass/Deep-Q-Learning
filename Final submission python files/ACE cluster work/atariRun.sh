#!/bin/bash
#SBATCH -n 8
#SBATCH -t 12:00:00
#SBATCH --mem=24G
source activate condaEnv
python3 nonJupyterDQN.py