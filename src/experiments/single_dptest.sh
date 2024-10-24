#!/bin/bash
run_path=$1 # Path storing checkpoints
ckpt_name=$2
head_name=$3
outdir="/mnt/workspace/public/multitask/eval_scripts/temp_frz_model/" # may need update
run_id=$(basename $run_path) # folder name as id
testfile="/mnt/workspace/public/multitask/eval_scripts/test_files/${head_name}.txt" # may need update

# freeze model
echo "Freezing Model -- $run_id -- $ckpt_name -- $head_name"
frozen_model="${run_id}#${ckpt_name}#${head_name}.pth"

cd $run_path
dp --pt freeze -o ${frozen_model} -c model.ckpt-${ckpt_name}.pt --head ${head_name}
if [ $? -ne 0 ]; then
  echo "Model Freezing Failed."
  exit 1
else
  echo "Model Freezing Finished."
fi

mv ${frozen_model} ${outdir}
cd $outdir

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
  echo "dp test result saved to ${outdir}${test_result}"
fi
