export netType='wide-resnet'
export depth=40
export width=10
export dataset='cifar10'
export save=logs/${dataset}/${netType}-${depth}x${width}/
export experiment_number=1
mkdir -p $save
mkdir -p modelState

th main.lua \
-dataset ${dataset} \
-netType ${netType} \
-nGPU 2 \
-batchSize 128 \
-top5_display false \
-testOnly false \
-dropout 0.3 \
-depth ${depth} \
-widen_factor ${width} \
| tee $save/train_log_${experiment_number}.txt
