

python pre_prepare_data.py --work_dir 'no-pe'
python ./pre_trainer.py --work_dir 'no-pe' --port 14521
python ./pre_evaluator.py --work_dir 'no-pe' --batch_size 1

python ./pre_trainer.py --work_dir 'you-pe' --port 14522
python ./pre_evaluator.py --work_dir 'you-pe' --batch_size 1











