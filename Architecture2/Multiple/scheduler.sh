#!/bin/sh

# Experimental auto scheduler to run parallel jobs. Inspired by launch.sh, but able to handles jobs
# launched at the same time

export CUDA_DEVICE_ORDER=PCI_BUS_ID  # because nvidia-smi is pants on head retarded
export MKL_NUM_THREADS=1  # For Anaconda MKL thing


{
    flock -x 1337

    if [ ! -f .cuda_list ]; then
        nvidia-smi --query-gpu=memory.free,index --format=csv,noheader,nounits | sort -nr | sed "s/^[0-9]+,[ \t]*//" -r > .cuda_list
    fi

    export CUDA_VISIBLE_DEVICES=`head -1 .cuda_list`
    tail -n +2 .cuda_list > .cuda_list
} 1337 > .cuda_list

"$@"
rm -f .cuda_list  # Then it will need to be re-generated, assume some other foreigner jobs might have finished