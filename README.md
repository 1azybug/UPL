# 项目上传github


**这里是ICAE 第二次前向不使用UPL的消融实验**

即,第一次前向**传入**`position_id`，第二次前向**不传入**`position_id`

注意:**所有操作和主分支是一样的**。

与主分支的区别是我们修改了代码：请看commit：`when use_pe is true, don't pass position_id in 2nd forward pass `

运行UPL的实验即可。

配置文件在`./experiment/ICAE_7B_UPL`

1. 项目文件夹简称UPL，首先安装python及其需要的包运行以下命令
```
git clone https://github.com/1azybug/UPL.git
cd UPL
conda create -n UPL python==3.10.4
conda activate UPL
pip install -r requirements.txt
```
2. 安装好环境后配置所需模型、数据集
-  模型：https://huggingface.co/meta-llama/Llama-2-7b-hf
-  预训练数据集：https://huggingface.co/datasets/DKYoon/SlimPajama-6B
-  微调数据集：https://huggingface.co/datasets/mrqa-workshop/mrqa

3. 切换到正确的分支
如果你想做ICAE的第二次前向的消融实验，忽略该步骤。

* 如果你想做其他的实验，可以用以下命令切换分支：
```
git branch -a
git checkout <分支名>
```



4. 修改experiment文件夹下的[config.json](./experiment/ICAE_7B_UPL/config.json)文件，填写模型和数据集的本地路径
```
"model_id": "your_model_path",
"dataset_repo": "your_data_path/DKYoon/SlimPajama-6B",
"instruction_dataset_repo": "your_data_path/mrqa-workshop_mrqa"
```

5. 修改experiment文件夹下的[config.json](./experiment/ICAE_7B_UPL/config.json)文件，**根据您的GPU数量修改梯度累积的步数**，确保 
```
batch_size_per_device*device_count*gradient_accumulation_steps==total_batch_size
```
```
"batch_size_per_device": 1,
"device_count": 8,
"gradient_accumulation_steps": 2,
```

6. 修改[pretrain_script.sh](./pretrain/pretrain_script.sh)和[sft_script.sh](./sft/sft_script.sh)里的`work_dir`参数。work_dir包含上面的config.json文件。
```
../experiment/ICAE_7B_UPL
```

7. 运行实验脚本
```
bash run.sh
```


