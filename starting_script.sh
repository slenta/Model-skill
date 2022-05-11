#!/bin/sh

#SBATCH -p prepost
#SBATCH --account=uo1075
#SBATCH --time=08:00:00



ly=$1

python save_lead_year_corr.py --lead_year ${ly}

