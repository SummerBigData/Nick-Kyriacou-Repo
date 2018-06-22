#PBS -N run_ex3
#PBS -1 walltime=10:00
#PBS -1 nodes=1:ppn=1
#PBS -1 mem=16GB
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
#
# This is the command line that runs the python script
python -u Neural_Networks_Learning.py >& output.log

