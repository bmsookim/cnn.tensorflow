export netType='resnet'
export depth=50
export width=2
export dataset='catdog'
export data='gen/catdog'
export save=logs/${dataset}/${netType}-${depth}x${width}/
export experiment_number=1
mkdir -p $save
mkdir -p modelState

th test.lua \
-dataset ${dataset} \
-data ${data} \
-netType ${netType} \
-resume modelState \
-nGPU 2 \
-batchSize 32 \
-dropout 0 \
-top5_display false \
-testOnly false \
-depth ${depth} \
-widen_factor ${width}
