
# python pre_prepare_data.py --work_dir '../experiment/local_experiment/ICAE_Llama-3.2-1B_DPL'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./pre_trainer.py --work_dir '../experiment/local_experiment/ICAE_Llama-3.2-1B_DPL' --port 14574
# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./pre_evaluator.py --work_dir '../experiment/local_experiment/ICAE_Llama-3.2-1B_DPL' --batch_size 1



# python pre_prepare_data.py --work_dir '../experiment/local_experiment/ICAE_Llama-3.2-1B_UPL'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./pre_trainer.py --work_dir '../experiment/local_experiment/ICAE_Llama-3.2-1B_UPL' --port 14574
# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./pre_evaluator.py --work_dir '../experiment/local_experiment/ICAE_Llama-3.2-1B_UPL' --batch_size 1

# python pre_prepare_data.py --work_dir '../experiment/local_experiment/ICAE_Llama-3.2-1B_DPL_single_chunk'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./pre_trainer.py --work_dir '../experiment/local_experiment/ICAE_Llama-3.2-1B_DPL_single_chunk' --port 14574
# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./pre_evaluator.py --work_dir '../experiment/local_experiment/ICAE_Llama-3.2-1B_DPL_single_chunk' --batch_size 1

# python pre_prepare_data.py --work_dir '../experiment/debug/500xquick'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./pre_trainer.py --work_dir '../experiment/debug/500xquick' --port 14574
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./pre_evaluator.py --work_dir '../experiment/debug/500xquick' --batch_size 1