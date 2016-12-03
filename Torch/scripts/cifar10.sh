export netType='wide-resnet'
export depth=40
export width=10
export dataset='cifar10'
export save=logs/${dataset}/${netType}-${depth}x${width}/
export experiment_number=1
# export CUDA_VISIBLE_DEVICE=0
mkdir -p $save
mkdir -p modelState

th main.lua \
-dataset ${dataset} \
-netType ${netType} \
-nGPU 2 \
-batchSize 128 \
-top5_display false \
-testOnly true \
-depth ${depth} \
-widen_factor ${width} \
| tee $save/test_log_${experiment_number}.txt
