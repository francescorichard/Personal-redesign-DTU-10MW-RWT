#!/bin/sh
#
#BSUB -J run_hawc2s
#BSUB -q hpc
#BSUB -n 1
#BSUB -W 03:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -o lsf_log/run_hawc2s_output_%J.out
#BSUB -e lsf_log/run_hawc2s_error_%J.err

# Check for HTCFILE
if [ -z "$HTCFILE" ]; then
    echo "ERROR: HTCFILE variable not defined."
    exit 1
fi

date
START_TIME=`date +%s`
echo "Running HAWC2S on: $HTCFILE"

module load hawcstab2

HAWC2S.exe "$HTCFILE"

END_TIME=`date +%s`
echo done
echo "runtime: $(($END_TIME-$START_TIME)) seconds"
