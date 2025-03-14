



# 运行前记得改一下learning_rate=0.00005,use_ae_loss=False

python instruction_prepare_data.py --work_dir 'no-pe'
python ./instruction_trainer.py --work_dir 'no-pe' --port 14523
python ./instruction_evaluator.py --work_dir 'no-pe' --batch_size 1

python ./instruction_trainer.py --work_dir 'you-pe' --port 14524
python ./instruction_evaluator.py --work_dir 'you-pe' --batch_size 1