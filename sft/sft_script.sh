




# 运行前记得改一下learning_rate=0.00005,use_ae_loss=False
python instruction_prepare_data.py --work_dir  '../experiment/no-pe'
python ./instruction_trainer.py --work_dir  '../experiment/no-pe' --port 14527
python ./instruction_evaluator.py --work_dir  '../experiment/no-pe' --batch_size 1


python ./instruction_trainer.py --work_dir  '../experiment/you-pe' --port 14528
python ./instruction_evaluator.py --work_dir  '../experiment/you-pe' --batch_size 1










