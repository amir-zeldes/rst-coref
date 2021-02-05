#!/bin/bash
python main.py \
	--train \
	--model_name "test_model.pt" \
	--model_type 0 \
	--train_dir "../data/train_dir" \
	--pretrained_coref_path ../pretrained_coref_model.pt
