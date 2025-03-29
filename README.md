# 注意力图分析实验

### 此部分为SFT阶段注意力分析图实验，运行方法和main以及其他分支相同。

### 你需要修改 model/modeling.py 中的 attn-analysis 里面的 save_dir 路径改为你的 work_dir 路径。

### 为了保证注意力图生成的唯一性，请用单gpu运行，具体指令如下

```
CUDA_VISIBLE_DEVICES=0 python ./instruction_evaluator.py --work_dir  '../experiment/ICAE' --batch_size 1
```