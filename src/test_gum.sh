#!/bin/bash
python main.py \
	--eval \
	--train_dir "../data/gum_train/" \
	--eval_dir "../data/gum_test/" \
	--model_name "gum_model.pt" \
	--model_type 0 \
	--pretrained_coref_path ../pretrained_coref_model.pt
