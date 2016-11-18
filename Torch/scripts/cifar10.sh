export netType='wide-resnet'
export dataset='cifar10'
export save=logs/${dataset}/${netType}/
export experiment_number=1
mkdir -p $save
th main.lua \
-dataset ${dataset} \
-netType ${netType} \
-nGPU 1 \
-batchSize 128 \
-top5_display false \
-depth 28 \
-widen_factor 10 \
| tee $save/train_log${experiment_number}.txt
