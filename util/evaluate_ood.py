# import json
# import sys
# import numpy as np
# import argparse
# import os

# parser = argparse.ArgumentParser()
# parser.add_argument("--work_dir", type=str, required=True, help="Directory containing the data files")
# args = parser.parse_args()

# input_path = os.path.join(args.work_dir, "output", "instruction_inference_results.json")
# output_path = os.path.join(args.work_dir, "output", "ood_subset_eval_results.json")

# sys.setrecursionlimit(10000)
# BioASQ_f1 = []
# DROP_f1 = []
# DuoRC_f1 = []
# RACE_f1 = []
# RelationExtraction_f1 = []
# TextbookQA_f1 = []
# total_f1 = []
# with open(input_path, 'r') as file:
#     data = json.load(file)
#     for example in data:
#         if example["subset"] == "BioASQ":
#             BioASQ_f1.append(example["rouge-f1"])
#             total_f1.append(example["rouge-f1"])
#         if example["subset"] == "DROP":
#             DROP_f1.append(example["rouge-f1"])
#             total_f1.append(example["rouge-f1"])
#         if example["subset"] == "DuoRC.ParaphraseRC":
#             DuoRC_f1.append(example["rouge-f1"])
#             total_f1.append(example["rouge-f1"])
#         if example["subset"] == "RACE":
#             RACE_f1.append(example["rouge-f1"])
#             total_f1.append(example["rouge-f1"])
#         if example["subset"] == "RelationExtraction":
#             RelationExtraction_f1.append(example["rouge-f1"])
#             total_f1.append(example["rouge-f1"])
#         if example["subset"] == "TextbookQA":
#             TextbookQA_f1.append(example["rouge-f1"])
#             total_f1.append(example["rouge-f1"])


# BioASQ_bleu4 = []
# DROP_bleu4 = []
# DuoRC_bleu4 = []
# RACE_bleu4 = []
# RelationExtraction_bleu4 = []
# TextbookQA_bleu4 = []
# total_bleu4 = []
# with open(input_path, 'r') as file:
#     data = json.load(file)
#     for example in data:
#         if example["subset"] == "BioASQ":
#             BioASQ_bleu4.append(example["bleu4"])
#             total_bleu4.append(example["bleu4"])
#         if example["subset"] == "DROP":
#             DROP_bleu4.append(example["bleu4"])
#             total_bleu4.append(example["bleu4"])
#         if example["subset"] == "DuoRC.ParaphraseRC":
#             DuoRC_bleu4.append(example["bleu4"])
#             total_bleu4.append(example["bleu4"])
#         if example["subset"] == "RACE":
#             RACE_bleu4.append(example["bleu4"])
#             total_bleu4.append(example["bleu4"])
#         if example["subset"] == "RelationExtraction":
#             RelationExtraction_bleu4.append(example["bleu4"])
#             total_bleu4.append(example["bleu4"])
#         if example["subset"] == "TextbookQA":
#             TextbookQA_bleu4.append(example["bleu4"])
#             total_bleu4.append(example["bleu4"])



#     print("BioASQ_f1:",np.mean(BioASQ_f1))
#     print("DROP_f1:",np.mean(DROP_f1))
#     print("DuoRC_f1:",np.mean(DuoRC_f1))
#     print("RACE_f1:",np.mean(RACE_f1))
#     print("RelationExtraction_f1:",np.mean(RelationExtraction_f1))
#     print("TextbookQA_f1:",np.mean(TextbookQA_f1))
#     print("total_f1:",np.mean(total_f1))
#     print("ood_test_samples_num:", len(total_f1))

#     print("BioASQ_bleu4:",np.mean(BioASQ_bleu4))
#     print("DROP_bleu4:",np.mean(DROP_bleu4))
#     print("DuoRC_bleu4:",np.mean(DuoRC_bleu4))
#     print("RACE_bleu4:",np.mean(RACE_bleu4))
#     print("RelationExtraction_bleu4:",np.mean(RelationExtraction_bleu4))
#     print("TextbookQA_bleu4:",np.mean(TextbookQA_bleu4))
#     print("total_bleu4:",np.mean(total_bleu4))
#     print("ood_test_samples_num:", len(total_bleu4))


# ood_results = {
#     "BioASQ_f1": np.mean(BioASQ_f1).item(),
#     "DROP_f1": np.mean(DROP_f1).item(),
#     "DuoRC_f1": np.mean(DuoRC_f1).item(),
#     "RACE_f1": np.mean(RACE_f1).item(),
#     "RelationExtraction_f1": np.mean(RelationExtraction_f1).item(),
#     "TextbookQA_f1": np.mean(TextbookQA_f1).item(),
#     "total_f1": np.mean(total_f1).item(),
#     "ood_test_samples_num_f1": len(total_f1),
#     "BioASQ_bleu4": np.mean(BioASQ_bleu4).item(),
#     "DROP_bleu4": np.mean(DROP_bleu4).item(),
#     "DuoRC_bleu4": np.mean(DuoRC_bleu4).item(),
#     "RACE_bleu4": np.mean(RACE_bleu4).item(),
#     "RelationExtraction_bleu4": np.mean(RelationExtraction_bleu4).item(),
#     "TextbookQA_bleu4": np.mean(TextbookQA_bleu4).item(),
#     "total_bleu4": np.mean(total_bleu4).item(),
#     "ood_test_samples_num_bleu4": len(total_bleu4)
# }

# with open(output_path, 'w') as json_file:
#     json.dump(ood_results, json_file, indent=4)


import json
import sys
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--work_dir", type=str, required=True, help="Directory containing the data files")
args = parser.parse_args()

input_path = os.path.join(args.work_dir, "output", "instruction_inference_results.json")
output_path = os.path.join(args.work_dir, "output", "ood_subset_eval_results.json")

sys.setrecursionlimit(10000)

# 初始化存储每个子集的F1, BLEU-4和EM分数的列表
metrics = {
    "BioASQ": {"f1": [], "bleu4": [], "em": []},
    "DROP": {"f1": [], "bleu4": [], "em": []},
    "DuoRC.ParaphraseRC": {"f1": [], "bleu4": [], "em": []},
    "RACE": {"f1": [], "bleu4": [], "em": []},
    "RelationExtraction": {"f1": [], "bleu4": [], "em": []},
    "TextbookQA": {"f1": [], "bleu4": [], "em": []}
}

total_f1 = []
total_bleu4 = []
total_em = []

with open(input_path, 'r') as file:
    data = json.load(file)
    for example in data:
        subset = example["subset"]
        if subset in metrics:
            metrics[subset]["f1"].append(example["rouge-f1"])
            metrics[subset]["bleu4"].append(example["bleu4"])
            # 假设数据中有一个字段"exact_match"用于表示EM分数
            metrics[subset]["em"].append(example["exact_match"])
            
            total_f1.append(example["rouge-f1"])
            total_bleu4.append(example["bleu4"])
            total_em.append(example["exact_match"])

# 计算每个子集的平均分数以及总平均分数
average_metrics = {}
for subset, scores in metrics.items():
    average_metrics[subset + "_f1"] = np.mean(scores["f1"]).item()
    average_metrics[subset + "_bleu4"] = np.mean(scores["bleu4"]).item()
    average_metrics[subset + "_em"] = np.mean(scores["em"]).item()

average_metrics["total_f1"] = np.mean(total_f1).item()
average_metrics["total_bleu4"] = np.mean(total_bleu4).item()
average_metrics["total_em"] = np.mean(total_em).item()
average_metrics["ood_test_samples_num"] = len(total_f1)

# 打印结果
for key, value in average_metrics.items():
    print(f"{key}: {value}")

# 将结果保存到JSON文件
with open(output_path, 'w') as json_file:
    json.dump(average_metrics, json_file, indent=4)
