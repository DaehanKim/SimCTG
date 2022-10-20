#!/bin/bash

# CUDA_VISIBLE_DEVICES=5 nohup python generate.py --model_path "/workspace/SimCTG/story_generation/simctg_cnndm_aug_min3_bsz16/training_step_30000_train_mle_loss_2.897_train_cl_loss_0.002_dev_ppl_17.362" \
# --gen_dir "generated_texts/simctg_cnndm/augmin3_bsz16" &> gen_augmin3_bsz16.out &
# CUDA_VISIBLE_DEVICES=6 nohup python generate.py  --model_path "/workspace/SimCTG/story_generation/simctg_cnndm_aug_min3_bsz256/training_step_14000_train_mle_loss_2.805_train_cl_loss_0.002_dev_ppl_16.991" \
# --gen_dir "generated_texts/simctg_cnndm/augmin3_bsz256" &> gen_augmin3_bsz256.out &
# CUDA_VISIBLE_DEVICES=7 nohup python generate.py  --model_path "/workspace/SimCTG/story_generation/simctg_cnndm_bt/training_step_29000_train_mle_loss_2.897_train_cl_loss_0.002_dev_ppl_18.475" \
# --gen_dir "generated_texts/simctg_cnndm/bt_wo_taskprefixvocab" --prefix "<_OT_>" &> gen_bt_wo_taskprefixvocab.out &
CUDA_VISIBLE_DEVICES=0 nohup python generate.py  --model_path "/workspace/SimCTG/story_generation/simctg_cnndm_bt_w_taskprefix/training_step_29000_train_mle_loss_2.894_train_cl_loss_0.002_dev_ppl_18.478" \
--gen_dir "generated_texts/simctg_cnndm/bt_w_taskprefixvocab" --prefix "<_OT_>" &> gen_bt_w_taskprefixvocab.out &
