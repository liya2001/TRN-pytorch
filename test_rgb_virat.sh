#!/usr/bin/env bash

#data-set
model_name="vehicle_person"
modality="RGB"

mkdir -p ./res/$model_name/$modality

python test_models.py $model_name $modality \
        ./models/vehicle_person/RGB/2018-08-27-19-31-06/TRN_vehicle_person_RGB_BNInception_TRNmultiscale_segment8_best.pth.tar \
        --num_class 11 \
        --val_list /home/ly/workspace/trecvid/data/person_vehicle_interaction/$model_name/label_file/val_$modality.txt \
        --root_path /home/ly/workspace/trecvid/data/person_vehicle_interaction/$model_name \
        --arch BNInception --test_segments 8 --gpus 0 -j 1 \
        --crop_fusion_type TRNmultiscale \
        --save_scores ./res/$model_name/$modality/vehicle_person_res.csv

