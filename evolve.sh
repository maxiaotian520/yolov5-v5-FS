#!/bin/bash
#python3 up_prune_v7_prune.py --data voc.yaml --cfg yolov5s.yaml --weights '' --sr 0.00001 --threshold $s

# for s in 0.08 0.085 0.09 0.095 0.1 0.105 0.11 0.115 0.12 0.125 0.13 0.135 0.14 0.145 0.15 0.155 0.16 0.165 0.17 0.175 0.18 0.185 0.19 0.195 0.2 0.205 0.21 0.215 0.22
# 	do
# 		python3 up_prune_v8_prune.py --data voc.yaml --cfg yolov5s.yaml --weights '' --sr 0.00001 --threshold $s
# 	done

python3 train_FS.py --epochs 10 --data coco.yaml --weights './weights/exp-COCO_0.99_0.01_01/weights/last.pt' --cfg yolov5s.yaml --batch-size 112 --scenario True --cache

python3 train_FS.py --epochs 10 --data coco.yaml --weights './weights/exp-COCO_0.99_0.01_01/weights/last.pt' --cfg yolov5s.yaml --batch-size 112 --evolve --cache
