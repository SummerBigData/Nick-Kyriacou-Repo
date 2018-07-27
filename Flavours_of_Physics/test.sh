#PBS -N run_ex3
#PBS -l walltime=600:00
#PBS -l nodes=1:ppn=1
#PBS -l mem=16GB
#PBS -j oe

# uncomment if using qsub
if [ -z "$PBS_O_WORKDIR"]
then
	echo "PBS_O_WORKDIR not defined"
else
	cd $PBS_O_WORKDIR
	echo $PBS_O_WORKDIR
fi
#
# Setup GPU code
module load python/2.7.8
source activate local
#
# This is the command line that runs the python script
python -u Data_Exploration.py >& making_plots_after.log

