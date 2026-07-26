#/bin/bash

# source ../envFlowDPS/bin/activate
# apt-get update && apt-get install -y libgl1

export CUDA=0

# deblur_motion
echo "开始执行任务: deblur_motion"
CUDA_VISIBLE_DEVICES=$CUDA python solve.py \
    --img_size 768 \
    --img_path data/AFHQ768_1000 \
    --workdir results/AFHQ-1000 \
    --prompt "a photo of a closed face of a dog" \
    --task deblur_motion \
    --method flowdps \
    --NFE 50 \
    --deg_scale 61 \
    --efficient_memory

# inpainting
echo "开始执行任务: inpainting"
CUDA_VISIBLE_DEVICES=$CUDA python solve.py \
    --img_size 768 \
    --img_path data/AFHQ768_1000 \
    --workdir results/AFHQ-1000 \
    --prompt "a photo of a closed face of a dog" \
    --task inpainting \
    --mask_type random \
    --method flowdps \
    --NFE 50 \
    --efficient_memory

# super_resolution
echo "开始执行任务: sr_bicubic"
CUDA_VISIBLE_DEVICES=$CUDA python solve.py \
    --img_size 768 \
    --img_path data/AFHQ768_1000 \
    --workdir results/AFHQ-1000 \
    --prompt "a photo of a closed face of a dog" \
    --task sr_bicubic \
    --method flowdps \
    --NFE 50 \
    --deg_scale 12 \
    --efficient_memory

# deblur_gauss
echo "开始执行任务: deblur_gauss"
CUDA_VISIBLE_DEVICES=$CUDA python solve.py \
    --img_size 768 \
    --img_path data/AFHQ768_1000 \
    --workdir results/AFHQ-1000 \
    --prompt "a photo of a closed face of a dog" \
    --task deblur_gauss \
    --method flowdps \
    --NFE 50 \
    --deg_scale 61 \
    --efficient_memory

# super_resolution
echo "开始执行任务: sr_avgpool"
CUDA_VISIBLE_DEVICES=$CUDA python solve.py \
    --img_size 768 \
    --img_path data/AFHQ768_1000 \
    --workdir results/AFHQ-1000 \
    --prompt "a photo of a closed face of a dog" \
    --task sr_avgpool \
    --method flowdps \
    --NFE 50 \
    --deg_scale 12 \
    --efficient_memory

METHODS=("flowdps")
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


# deblur_motion
echo "开始执行任务: deblur_motion"
CUDA_VISIBLE_DEVICES=$CUDA python solve.py \
    --img_size 768 \
    --img_path data/FFHQ768_1000 \
    --workdir results/FFHQ-1000 \
    --prompt "a photo of a closed face" \
    --task deblur_motion \
    --method flowdps \
    --NFE 50 \
    --deg_scale 61 \
    --efficient_memory

# inpainting
echo "开始执行任务: inpainting"
CUDA_VISIBLE_DEVICES=$CUDA python solve.py \
    --img_size 768 \
    --img_path data/FFHQ768_1000 \
    --workdir results/FFHQ-1000 \
    --prompt "a photo of a closed face" \
    --task inpainting \
    --mask_type random \
    --method flowdps \
    --NFE 50 \
    --efficient_memory

# super_resolution
echo "开始执行任务: sr_bicubic"
CUDA_VISIBLE_DEVICES=$CUDA python solve.py \
    --img_size 768 \
    --img_path data/FFHQ768_1000 \
    --workdir results/FFHQ-1000 \
    --prompt "a photo of a closed face" \
    --task sr_bicubic \
    --method flowdps \
    --NFE 50 \
    --deg_scale 12 \
    --efficient_memory

# deblur_gauss
echo "开始执行任务: deblur_gauss"
CUDA_VISIBLE_DEVICES=$CUDA python solve.py \
    --img_size 768 \
    --img_path data/FFHQ768_1000 \
    --workdir results/FFHQ-1000 \
    --prompt "a photo of a closed face" \
    --task deblur_gauss \
    --method flowdps \
    --NFE 50 \
    --deg_scale 61 \
    --efficient_memory

# super_resolution
echo "开始执行任务: sr_avgpool"
CUDA_VISIBLE_DEVICES=$CUDA python solve.py \
    --img_size 768 \
    --img_path data/FFHQ768_1000 \
    --workdir results/FFHQ-1000 \
    --prompt "a photo of a closed face" \
    --task sr_avgpool \
    --method flowdps \
    --NFE 50 \
    --deg_scale 12 \
    --efficient_memory

METHODS=("flowdps")
TASKS=("sr_avgpool" "deblur_gauss" "deblur_motion" "inpainting_random" "sr_bicubic" )
for METHOD in "${METHODS[@]}"; do
    for TASK in "${TASKS[@]}"; do
        echo "开始计算指标：$METHOD $TASK"
        CUDA_VISIBLE_DEVICES=$CUDA python eval.py \
            --path2="results/FFHQ-1000/$METHOD/$TASK/recon" \
            --path1="data/FFHQ768_1000" \
            --save_file="results/FFHQ-1000/$METHOD/$TASK/metrics.txt" \
            --metric psnr ssim fid lpips
    done
done


# deblur_motion
echo "开始执行任务: deblur_motion"
CUDA_VISIBLE_DEVICES=$CUDA python solve.py \
    --img_size 768 \
    --img_path data/DIV2K_800 \
    --workdir results/DIV2K-800 \
    --prompt_file data/DIV2K_800_prompts.txt \
    --task deblur_motion \
    --method flowdps \
    --NFE 50 \
    --deg_scale 61 \
    --efficient_memory

# inpainting
echo "开始执行任务: inpainting"
CUDA_VISIBLE_DEVICES=$CUDA python solve.py \
    --img_size 768 \
    --img_path data/DIV2K_800 \
    --workdir results/DIV2K-800 \
    --prompt_file data/DIV2K_800_prompts.txt \
    --task inpainting \
    --mask_type random \
    --method flowdps \
    --NFE 50 \
    --efficient_memory

# super_resolution
echo "开始执行任务: sr_bicubic"
CUDA_VISIBLE_DEVICES=$CUDA python solve.py \
    --img_size 768 \
    --img_path data/DIV2K_800 \
    --workdir results/DIV2K-800 \
    --prompt_file data/DIV2K_800_prompts.txt \
    --task sr_bicubic \
    --method flowdps \
    --NFE 50 \
    --deg_scale 12 \
    --efficient_memory

# deblur_gauss
echo "开始执行任务: deblur_gauss"
CUDA_VISIBLE_DEVICES=$CUDA python solve.py \
    --img_size 768 \
    --img_path data/DIV2K_800 \
    --workdir results/DIV2K-800 \
    --prompt_file data/DIV2K_800_prompts.txt \
    --task deblur_gauss \
    --method flowdps \
    --NFE 50 \
    --deg_scale 61 \
    --efficient_memory

# super_resolution
echo "开始执行任务: sr_avgpool"
CUDA_VISIBLE_DEVICES=$CUDA python solve.py \
    --img_size 768 \
    --img_path data/DIV2K_800 \
    --workdir results/DIV2K-800 \
    --prompt_file data/DIV2K_800_prompts.txt \
    --task sr_avgpool \
    --method flowdps \
    --NFE 50 \
    --deg_scale 12 \
    --efficient_memory

METHODS=("flowdps") 
TASKS=("sr_avgpool" "deblur_gauss" "deblur_motion" "inpainting_random" "sr_bicubic" )
for METHOD in "${METHODS[@]}"; do
    for TASK in "${TASKS[@]}"; do
        echo "开始计算指标：$METHOD $TASK"
        CUDA_VISIBLE_DEVICES=$CUDA python eval.py \
            --path2="results/DIV2K-800/$METHOD/$TASK/recon" \
            --path1="data/DIV2K_800" \
            --save_file="results/DIV2K-800/$METHOD/$TASK/metrics.txt" \
            --metric psnr ssim fid lpips
    done
done