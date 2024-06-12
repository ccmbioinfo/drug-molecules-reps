sbatch -N 1 -c 12 --mem 100G --time $3:00:00  --job-name $2 --tmp=300G $1
