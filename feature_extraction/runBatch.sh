if [ -n "$4" ]; then
    DEPENDENCY_ARG="--dependency=afterok:$4"
else
    DEPENDENCY_ARG=""
fi

sbatch -N 1 -c 20 --mem 100G $DEPENDENCY_ARG --time $3:00:00  --job-name $2 --tmp=300G $1
#sbatch -N 1 -c 20 --mem 100G --dependency=afterok:$4 --time $3:00:00  --job-name $2 --tmp=300G $1
