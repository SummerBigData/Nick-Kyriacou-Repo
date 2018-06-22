#PBS -N run_ex3
#PBS -l walltime=180:00
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
#
# This is the command line that runs the python script
python -u Sparse_Auto_Encoder_Color_Features.py >& minimizing_cost_function_using_their_values.log

