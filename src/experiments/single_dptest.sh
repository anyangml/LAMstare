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
if [ ! -f $frozen_model ]; then
  dp --pt freeze -o ${frozen_model} -c model.ckpt-${ckpt_name}.pt --head ${head_name}
  if [ $? -ne 0 ]; then
    echo "Model Freezing Failed."
    exit 1
  else
    echo "Model Freezing Finished."
  fi
else
  echo "Model Already Freezed."
fi


mv ${frozen_model} ${outdir}
cd $outdir

test_result=${run_id}#${ckpt_name}#${head_name}.txt
if [ ! -f "${frozen_model}" ]; then
  echo "No Frozen Model ${frozen_model}."
  exit 1
fi
if [ ! -f ${test_result} ]; then
  dp --pt test -m ${frozen_model} -f ${testfile} -l ${test_result}
else
  echo "Test Result Already Exists." # FIXME: need to determine if test finishes correctly
fi
echo "dp test result saved to ${outdir}${test_result}"
