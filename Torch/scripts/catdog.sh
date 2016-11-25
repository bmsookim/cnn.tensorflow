export netType='wide-resnet'
export dataset='imagenet'
export data='preprocessedData'
export save=logs/catdog/${netType}
export experiment_numer=2
mkdir -p $save
th main.lua \
-dataset ${dataset} \
-data ${data} \
-netType ${netType} \
-nGPU 2 \
-batchSize 32 \
-nClasses 2 \
-resetClassifier true \
-top5_display false \
-depth 50 \
-widen_factor 1 \
| tee $save/train_log${experiment_numebr}.txt
