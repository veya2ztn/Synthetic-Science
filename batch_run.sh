CPU_NUM=64 # Automatically get the number of CPUs
for ((CPU=0; CPU<CPU_NUM; CPU++));
do
sbatch --quotatype=spot -p AI4Phys -N1 -c4 --gres=gpu:1  run.sh data/archive_md.filelist $CPU $CPU_NUM
done 
