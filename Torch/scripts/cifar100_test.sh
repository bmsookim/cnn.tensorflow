export netType='wide-resnet'
export depth=40
export width=10
export dataset='cifar100'
export save=logs/${dataset}/${netType}-${depth}x${width}/
export experiment_number=1
export CUDA_VISIBLE_DEVICES="0, 1"
mkdir -p $save
mkdir -p modelState

th main.lua \
-dataset ${dataset} \
-netType ${netType} \
-nGPU 2 \
-top5_display true \
-testOnly true \
-dropout 0.3 \
-batchSize 128 \
-depth ${depth} \
-widen_factor ${width} \
| tee $save/test_log_${experiment_number}.txt
