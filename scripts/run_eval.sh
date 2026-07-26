#/bin/bash

# source ../envFlowDPS/bin/activate
# apt-get update && apt-get install -y libgl1

export CUDA=0


METHODS=("latentdaps") 
TASKS=("sr_avgpool" "deblur_gauss" "deblur_motion" "inpainting_random" "sr_bicubic")

for METHOD in "${METHODS[@]}"; do
    for TASK in "${TASKS[@]}"; do
        echo "开始计算指标：$METHOD $TASK"
        CUDA_VISIBLE_DEVICES=$CUDA python eval.py \
            --path2="results/AFHQ-1000/$METHOD/$TASK/recon" \
            --path1="data/AFHQ768_1000" \
            --save_file="results/AFHQ-1000/$METHOD/$TASK/metrics.txt" \
            --metric psnr ssim fid lpips
    done
done


