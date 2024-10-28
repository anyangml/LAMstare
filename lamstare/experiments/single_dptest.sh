#!/bin/bash

run_id="$1"
ckpt_name=$2
head_name=$3
prefix="/mnt/workspace/public/multitask/training_exps/" # may need update
outdir="/mnt/workspace/public/multitask/eval_scripts/temp_frz_model" # may need update
run_pth="${prefix}${run_id}"
testfile="/mnt/workspace/public/multitask/eval_scripts/test_files/${head_name}.txt" # may need update



# freeze model
echo "Freezing Model -- $run_id -- $ckpt_name -- $head_name"
echo ""
cd $run_pth
dp --pt freeze -o ${run_id}#${ckpt_name}#${head_name}.pth -c model.ckpt-${ckpt_name}.pt --head ${head_name}

if [ $? -ne 0 ]; then
  echo "Model Freezing Failed."
  exit 1
else
  echo "Model Freezing Finished."
fi

mv ${run_id}#${ckpt_name}#${head_name}.pth ${outdir}
cd $outdir

if [ -f "${run_id}#${ckpt_name}#${head_name}.pth" ]; then
  dp --pt test -m ${run_id}#${ckpt_name}#${head_name}.pth -f ${testfile} -l ${run_id}#${ckpt_name}#${head_name}.txt
else
  echo "No Freezed Model."
  exit 1
fi
