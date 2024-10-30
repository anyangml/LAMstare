#!/bin/bash -x
exp_path=$1 # Path storing checkpoints
ckpt_name=$2
head_name=$3 # perform single task test if head name not given
temp_file_path=$4
run_id=$(basename $exp_path) # folder name as id
testfile="${temp_file_path}${run_id}#${ckpt_name}#${head_name}_valid.txt"  # contains paths to test sets

# freeze model
echo "Freezing Model -- $run_id -- $ckpt_name -- $head_name"
frozen_model="${run_id}#${ckpt_name}#${head_name}.pth"

cd $exp_path
# omit --head for single task
dp --pt freeze -o ${frozen_model} -c model.ckpt-${ckpt_name}.pt ${head_name:+--head ${head_name}}
if [ $? -ne 0 ]; then
  echo "Model Freezing Failed."
  exit 1
else
  echo "Model Freezing Finished."
fi

mv ${frozen_model} ${temp_file_path}
cd $temp_file_path

test_result=${run_id}#${ckpt_name}#${head_name}.txt
if [ ! -f "${frozen_model}" ]; then
  echo "No Frozen Model ${frozen_model}."
  exit 1
else
  dp --pt test -m ${frozen_model} -f ${testfile} -l ${test_result}
fi
if [ $? -ne 0 ]; then
  echo "dp test Failed!"
  exit 1
else
  echo "dp test result saved to ${temp_file_path}${test_result}"
fi
