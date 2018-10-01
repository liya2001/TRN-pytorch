#!/usr/bin/env bash

#data-set
model_name="vehicle_person"
modality="RGB"
DATE=`date '+%Y-%m-%d-%H-%M-%S'`

mkdir -p ./nohups/$model_name/$modality
mkdir -p ./models/$model_name/$modality/$DATE

nohup python main.py $model_name $modality \
        --num_class 11 \
        --train_list /home/ly/workspace/trecvid/data/person_vehicle_interaction/$model_name/label_file/train_$modality.txt \
        --val_list /home/ly/workspace/trecvid/data/person_vehicle_interaction/$model_name/label_file/val_$modality.txt \
        --root_path /home/ly/workspace/trecvid/data/person_vehicle_interaction/$model_name \
        --pretrained /home/ly/workspace/trecvid/pretrained_models/trn/TRN_somethingv2_RGB_BNInception_TRNmultiscale_segment8_best.pth.tar \
        --arch BNInception --num_segments 8 --gpus 0 1 \
        --lr_steps 50 100 --epochs 120 \
        -b 64 -j 24 \
        --consensus_type TRNmultiscale \
        --root_model ./models/$model_name/$modality/$DATE \
> nohups/$model_name/$modality/${DATE}-log.out 2>&1 &
