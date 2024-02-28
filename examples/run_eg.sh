# direct
python main-0420.py --dataset hotpot --task step1 --split dev --promptfile xx --pid xx --output_dir xx
python main-0420.py --dataset hotpot --task wiki --split dev --promptfile xx --pid xx  --search bing --output_dir xx

# LLM rewrite 
python main-0420.py --dataset hotpot --task rewrite --split dev --promptfile xx --pid xx  --search bing --output_dir xx --think
python main-0420.py --dataset hotpot --task rewrite2 --split dev --pid xx --promptfile xx --search bing  --repid xx --output_dir xx 

python main-0514.py --dataset mmlusocial --task rewrite --split test --promptfile xx --pid xx  --search bing --output_dir xx --retrieve plain --nums 1000
python main-0514.py --dataset mmlusocial --task rewrite2 --split test --pid xx --promptfile xx --search bing  --repid xx --output_dir xx --nums 1000

# trainable rewrite
CUDA_VISIBLE_DEVICES=15,14,13,12,11,10,9,8,7,6 python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/hotpot/t5_ppo_0314_v2.yml --experiment_name xx
CUDA_VISIBLE_DEVICES=0,1,2,3,4,10,11,12,13,14 python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/hotpot/t5_ambig_0525_v3.yml --experiment_name xx
CUDA_VISIBLE_DEVICES=5,6,7,8,9,10,11,12,15,14 python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/hotpot/other_0610.yml  --experiment_name xx