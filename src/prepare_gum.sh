#!/bin/bash
python main.py \
	--prepare \
	--train_dir ../data/gum_train \
	--pretrained_coref_path ../pretrained_coref_model.pt
