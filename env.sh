# Add any `module load` or `export` commands that your code needs to
# compile and run to this file.
module load GCC/7.2.0-2.29
module load slurm-21.08
export OMP_NUM_THREADS=28
export OMP_PLACES=cores
