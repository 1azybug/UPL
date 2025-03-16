# 项目上传github
1. 项目文件夹简称UPL，首先安装python及其需要的包运行以下命令
```
git clone https://github.com/lx-Meteors/More-effective-context-compression-using-compressed-tokens.git
cd More-effective-context-compression-using-compressed-tokens
conda create -n UPL python==3.10.4
conda activate UPL
pip install -r requirements.txt
```
2. 安装好环境后配置所需模型、数据集
-  模型：https://huggingface.co/meta-llama/Llama-2-7b-hf
-  预训练数据集：https://huggingface.co/datasets/DKYoon/SlimPajama-6B
-  微调数据集：https://huggingface.co/datasets/mrqa-workshop/mrqa

3. 修改experiment文件夹下的[config.json](./experiment/main/ICAE_1.1B_UPL/config.json)文件，填写模型和数据集的本地路径
```
"model_id": "your_model_path",
```

4. 修改experiment文件夹下的[config.json](./experiment/main/ICAE_1.1B_UPL/config.json)文件，**根据您的GPU数量修改梯度累积的步数**，确保 
```
batch_size_per_device*device_count*gradient_accumulation_steps==total_batch_size
```
```
"batch_size_per_device": 1,
"device_count": 8,
"gradient_accumulation_steps": 2,
```

在训练之前，在experiment文件夹下创建 sy_1文件夹，在 sy_1下添加config配置模型和数据集路径以及超参数
- 若想去掉位置编码，则直接在config中use_pe=False
- 若想去掉ae_loss，则直接在config中use_ae_loss=False

## 继续预训练
```
注意事项：config的learning_rate是0.0001
cd pretrain

处理一次数据集即可：python pre_prepare_data.py --work_dir '../experiment/example'
训练模型：python ./pre_trainer.py --work_dir '../experiment/example' --port 14529
测试模型：python ./pre_evaluator.py --work_dir '../experiment/example' --batch_size 1
```

## 微调
```
注意事项：请修改config的learning_rate为0.00005
记得在config去掉ae_loss
cd sft
处理一次数据集即可：python instruction_prepare_data.py --work_dir '../experiment/sy_1'

微调训练：python ./instruction_trainer.py --work_dir '../experiment/sy_1' --port 14525
模型测试：python ./instruction_evaluator.py --work_dir '../experiment/sy_1' --batch_size 1
训练模型和测试结果均保存在sy_1文件夹里


获取6个数据集的分别指标：
cd util
更改目录为当前结果 sy_1 运行evaluate_ood.py
```

## 500xCompress复现
```
配一下config的数据集和模型路径
cd 500xCompress
预训练...
微调...
```

# 节省时间简化版
```
配置好每个config的模型和数据集路径
500xCompress复现
cd 500xCompress
bash pretrain.sh 等待完事即可
bash sft.sh 等待完事即可

icae的复现
cd pretrain
bash pretrain_script.sh 等待完事即可
bash sft_script.sh 等待完事即可

最后进入util的evaluate_ood，配一下每个config的路径，测sft结果
```
