CUDA_VISIBLE_DEVICES=9,10,11,12,13,14,15 python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/tqa/t5_ppo_0314_v3.yml --experiment_name 4beam_de2

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/hotpot/t5_ppo_0314_v3.yml --experiment_name flant5l-turbo-hotpot-0404-de

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9,10,11 python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/hotpot/flant5_ppo_0403_v3.yml --experiment_name flant5l-turbo-hotpot-0404
CUDA_VISIBLE_DEVICES=15,14,13,12,11,10,9,8,7,6 python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/hotpot/t5_ppo_0314_v2.yml --experiment_name t5l-turbo-hotpot-0330
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/hotpot/flant5_ppo_0403_v3.yml --experiment_name flant5l-turbo-hotpot-0405-6k

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9,10,11 python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/hotpot/flant5_ppo_0403_v3.yml --experiment_name flant5l-turbo-hotpot-0407-6k

# chinese
CUDA_VISIBLE_DEVICES=5,6,7,8,9,10,11,12,13,14 python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/hotpot/mt5_jec_v2.yml --experiment_name mt5_jec
CUDA_VISIBLE_DEVICES=0,1,2,3,4,10,11,12,13,14 python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/hotpot/mt5b_jec_v2.yml --experiment_name mt5b_jec
CUDA_VISIBLE_DEVICES=0,1,2,3,4,10,11,12,13,14 python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/hotpot/rdt5_jec.yml --experiment_name rdt5_jec

# ambig
CUDA_VISIBLE_DEVICES=0,1,2,3,4,10,11,12,13,14 python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/hotpot/t5_ambig_0525_v3.yml --experiment_name t5_ambig
CUDA_VISIBLE_DEVICES=5,6,7,8,9,10,11,12,13,14 python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/hotpot/t5_ambig_0525_v3.yml --experiment_name t5_ambig_27

CUDA_VISIBLE_DEVICES=5,6,7,8,9,10,11,12,13,14 python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/hotpot/t5_ambig_debug_v3.yml --experiment_name t5_ambig_27

# popqa
CUDA_VISIBLE_DEVICES=0,1,2,3,4,10,11,12,13,14 python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/hotpot/t5_pop_0609_v3.yml --experiment_name t5_popqa_debug
CUDA_VISIBLE_DEVICES=0,1,2,3,4,10,11,12,13,14 python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/hotpot/t5_pop_0609_v3.yml --experiment_name t5_popqa
CUDA_VISIBLE_DEVICES=0,1,2,3,4,10,11,12,13,14 python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/hotpot/t5_pop_0610_v3.yml --experiment_name t5_popqa_real0610

CUDA_VISIBLE_DEVICES=0,1,2,3,4,10,11,12,13,14 python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/hotpot/t5_pop_0612_v3.yml --experiment_name t5_popqa_0612

#mmlu
CUDA_VISIBLE_DEVICES=5,6,7,8,9,10,11,12,15,14 python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/hotpot/other_0610.yml  --experiment_name other_0610

CUDA_VISIBLE_DEVICES=5,6,7,8,9,10,11,12,15,14 python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/hotpot/social_0615.yml  --experiment_name social_0615
CUDA_VISIBLE_DEVICES=4,3,2,5,6,7,10,11,12,13,14,1,0 python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/hotpot/human_0618.yml  --experiment_name human_0618

CUDA_VISIBLE_DEVICES=4,3,2,5,6,7,10,11,12,13,14,1,0 python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/hotpot/stem_0618.yml  --experiment_name stem_0618

CUDA_VISIBLE_DEVICES=15,14,13,12,11,10,9,8,7,6 python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/hotpot/t5_ppo_0314_v2.yml --experiment_name t5l-turbo-hotpot-0331

