

python ./instruction_trainer.py --work_dir 'instruction_rank-128_lm_cl_mrqa_param_eq' --port 14522
python ./instruction_trainer.py --work_dir 'pre-train-q-v-mrqa-decoder-wo_pi-lm-cl' --port 14523
python ./instruction_evaluator.py --work_dir 'in-domain-tinyllama_instruction' --batch_size 1


python ./instruction_evaluator.py --work_dir 'in-domain-pre-train-q-v-mrqa-lm' --batch_size 1
python ./instruction_evaluator.py --work_dir 'in-domain-pre-train-q-v-mrqa-lm-cl' --batch_size 1
# 修改decoder
python ./instruction_evaluator.py --work_dir 'in-domain-pre-train-q-v-mrqa-decoder-lm' --batch_size 1
python ./instruction_evaluator.py --work_dir 'in-domain-pre-train-q-v-mrqa-decoder-lm-cl' --batch_size 1

python ./instruction_evaluator.py --work_dir 'in-domain-pre-train-q-v-mrqa-decoder-wo_pi-lm' --batch_size 1
CUDA_VISIBLE_DEVICES=0,1,2,6 python ./instruction_evaluator.py --work_dir 'new-you-pe-cl' --batch_size 1


CUDA_VISIBLE_DEVICES=1,2,3,4 nohup python ./pre_trainer.py --work_dir 'you-pe' --port 14529
nohup python ./instruction_trainer.py --work_dir 'new-you-pe-cl' --port 14524

nohup python ./pre_trainer.py --work_dir 'new-you-pe-cl' --port 14529
#python ./pre_evaluator.py --work_dir 'pre-train-q-v-mrqa-decoder-ae-lm' --batch_size 1











