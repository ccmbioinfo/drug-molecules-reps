#!/bin/bash


# Function to append ':00:00' to hour argument if needed
append_time_format() {
    if [[ $3 =~ ^[0-9]+$ ]]; then
        echo "$3:00:00"
    else
        echo "$3"
    fi
}

# Append ':00:00' to hour argument if needed
time_arg=$(append_time_format "$3")
job_name=$2

#SBATCH --job-name=$job_name                # Job name
#SBATCH --time=$time_arg                   # Time limit hrs:min:sec
#SBATCH --ntasks=1                       # Number of tasks (usually set to 1 for serial jobs)
#SBATCH --mem=100G
#SBATCH --cpus-per-task=20 

if [ "$4" ]; then
    echo "#SBATCH --dependency=afterok:$4"
fi

python feature_extraction_DUD.py


echo "Job submitted!"

