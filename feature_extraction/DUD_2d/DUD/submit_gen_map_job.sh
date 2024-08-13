#!/bin/bash

shift

# Function to append ':00:00' to hour argument if needed
append_time_format() {
    if [[ $1 =~ ^[0-9]+$ ]]; then
        echo "$1:00:00"
    else
        echo "$1"
    fi
}

# Append ':00:00' to hour argument if needed
time_arg=$(append_time_format "$1")

#SBATCH --job-name="$1"                # Job name
#SBATCH --time=$time_arg                   # Time limit hrs:min:sec
#SBATCH --ntasks=1                       # Number of tasks (usually set to 1 for serial jobs)
#SBATCH --mem=100G
#SBATCH --cpus-per-task=20 

if [ "$3" ]; then
    echo "#SBATCH --dependency=afterok:$3"
fi

python generate_umap_dims_DUD.py


echo "Job submitted!"
