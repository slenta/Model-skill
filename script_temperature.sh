#!/bin/bash

#SBATCH -J SimonLentz
#SBATCH -p compute
#SBATCH --account=uo1075
#SBATCH --time=08:00:00
#SBATCH --mem=400G
#SBATCH --output=my_temperature_job.o%j 

module load python3/2022.01-gcc-11.2.0

lead_year=$1
model=$2

python 4d_temperature_lys.py --lead_year $lead_year --start_year 1960 --end_year 2019 --region 'global' MPI 




