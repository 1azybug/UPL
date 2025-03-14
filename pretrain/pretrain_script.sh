
python pre_prepare_data.py --work_dir '../experiment/no-pe'
python ./pre_trainer.py --work_dir '../experiment/no-pe' --port 14525
python ./pre_evaluator.py --work_dir '../experiment/no-pe' --batch_size 1

python ./pre_trainer.py --work_dir '../experiment/you-pe' --port 14526
python ./pre_evaluator.py --work_dir '../experiment/you-pe' --batch_size 1











