# !/bin/bash

gpu_ids="2"
export CUDA_VISIBLE_DEVICES=$gpu_ids

train_script="train.py"

# log file
log_file="111_edm.log"

# cmd="nohup python ${train_script} > ${log_file} 2>&1 &"
cmd="python ${train_script}"
eval ${cmd}

# PID 写入
echo $! >> output.log

echo "Running command: ${cmd} ON GPU ${gpu_ids}"