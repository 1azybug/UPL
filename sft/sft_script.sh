

python instruction_prepare_data.py --work_dir  '../experiment/local_experiment/ICAE_Llama-3.2-1B_DPL'
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./instruction_trainer.py --work_dir  '../experiment/local_experiment/ICAE_Llama-3.2-1B_DPL' --port 14527
# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./instruction_evaluator.py --work_dir  '../experiment/no-pe' --batch_size 1


# python ./instruction_trainer.py --work_dir  '../experiment/you-pe' --port 14528
# python ./instruction_evaluator.py --work_dir  '../experiment/you-pe' --batch_size 1










