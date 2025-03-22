# python instruction_prepare_data.py --work_dir  '../experiment/local_experiment/ICAE_Llama-3.2-1B_DPL'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./instruction_trainer.py --work_dir  '../experiment/local_experiment/ICAE_Llama-3.2-1B_DPL' --port 14527
# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./instruction_evaluator.py --work_dir  '../experiment/local_experiment/ICAE_Llama-3.2-1B_DPL' --batch_size 1
# python ../util/evaluate_ood.py --work_dir '../experiment/local_experiment/ICAE_Llama-3.2-1B_DPL'
# python ../util/evaluate_iid.py --work_dir '../experiment/local_experiment/ICAE_Llama-3.2-1B_DPL'

# python instruction_prepare_data.py --work_dir  '../experiment/local_experiment/ICAE_Llama-3.2-1B_UPL'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./instruction_trainer.py --work_dir  '../experiment/local_experiment/ICAE_Llama-3.2-1B_UPL' --port 14527
# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./instruction_evaluator.py --work_dir  '../experiment/local_experiment/ICAE_Llama-3.2-1B_UPL' --batch_size 1
# python ../util/evaluate_ood.py --work_dir '../experiment/local_experiment/ICAE_Llama-3.2-1B_UPL'
# python ../util/evaluate_iid.py --work_dir '../experiment/local_experiment/ICAE_Llama-3.2-1B_UPL'


python instruction_prepare_data.py --work_dir  '../experiment/debug/quick'
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./instruction_trainer.py --work_dir  '../experiment/debug/quick' --port 14527
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./instruction_evaluator.py --work_dir  '../experiment/debug/quick' --batch_size 1
python ../util/evaluate_ood.py --work_dir '../experiment/debug/quick'
python ../util/evaluate_iid.py --work_dir '../experiment/debug/quick'





