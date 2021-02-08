#!/bin/bash
python main.py \
	--eval \
	--train_dir "../data/train_dir" \
	--eval_dir "../data/test_dir/" \
	--model_name "test_model.pt" \
	--model_type 0 \
	--pretrained_coref_path ../pretrained_coref_model.pt
