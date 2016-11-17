export netType='wide-resnet'
export dataset='cifar100'
export save=logs/${dataset}/${netType}/
export experiment_number=1
mkdir -p $save
th main.lua \
-dataset ${dataset} \
-netType ${netType} \
-nGPU 2 \
-top5_display true \
-batchSize 128 \
-depth 28 \
-widen_factor 10 \
| tee $save/train_log${experiment_number}.txt
