#!/usr/bin/env bash


CUDA_VISIBLE_DEVICES=2, python ./main_sweep.py --config ./configs/sweep_jsons/radar_sweep_resnet.json &
sleep 5
CUDA_VISIBLE_DEVICES=3, python ./main_sweep.py --config ./configs/sweep_jsons/radar_sweep_cnn.json &