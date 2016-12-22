#!/bin/bash
export netType='wide-resnet'
export depth=40
export width=2
export dataset='catdog'
export data='gen/catdog'
export save=logs/${dataset}/${netType}-${depth}x${width}/
export experiment_number=1
mkdir -p $save
mkdir -p modelState

th main.lua \
    -dataset ${dataset} \
    -data ${data} \
    -netType ${netType} \
    -nGPU 2 \
    -batchSize 32 \
    -display_iter 100 \
    -top5_display false \
    -testOnly false \
    -depth ${depth} \
    -widen_factor ${width} \
    | tee $save/train_log_${experiment_number}.txt

th convert.lua \
    -dataset ${dataset} \
    -data ${data} \
    -netType ${netType} \
    -nGPU 2 \
    -batchSize 32 \
    -depth ${depth} \
    -widen_factor ${width} \
    -optnet true \
    | tee $save/convert_log.txt
