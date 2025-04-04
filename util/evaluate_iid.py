# import json
# import sys
# import numpy as np
# import argparse
# import os

# parser = argparse.ArgumentParser()
# parser.add_argument("--work_dir", type=str, required=True, help="Directory containing the data files")
# args = parser.parse_args()

# input_path = os.path.join(args.work_dir, "output", "instruction_inference_results.json")
# output_path = os.path.join(args.work_dir, "output", "iid_subset_eval_results.json")

# sys.setrecursionlimit(10000)
# SQuAD_f1 = []
# NewsQA_f1 = []
# TriviaQA_f1 = []
# SearchQA_f1 = []
# HotpotQA_f1 = []
# NaturalQuestions_f1 = []
# total_f1 = []
# with open(input_path, 'r') as file:
#     data = json.load(file)
#     for example in data:
#         if example["subset"] == "SQuAD":
#             SQuAD_f1.append(example["rouge-f1"])
#             total_f1.append(example["rouge-f1"])
#         if example["subset"] == "NewsQA":
#             NewsQA_f1.append(example["rouge-f1"])
#             total_f1.append(example["rouge-f1"])
#         if example["subset"] == "TriviaQA-web":
#             TriviaQA_f1.append(example["rouge-f1"])
#             total_f1.append(example["rouge-f1"])
#         if example["subset"] == "SearchQA":
#             SearchQA_f1.append(example["rouge-f1"])
#             total_f1.append(example["rouge-f1"])
#         if example["subset"] == "HotpotQA":
#             HotpotQA_f1.append(example["rouge-f1"])
#             total_f1.append(example["rouge-f1"])
#         if example["subset"] == "NaturalQuestionsShort":
#             NaturalQuestions_f1.append(example["rouge-f1"])
#             total_f1.append(example["rouge-f1"])


# SQuAD_bleu4 = []
# NewsQA_bleu4 = []
# TriviaQA_bleu4 = []
# SearchQA_bleu4 = []
# HotpotQA_bleu4 = []
# NaturalQuestions_bleu4 = []
# total_bleu4 = []
# with open(input_path, 'r') as file:
#     data = json.load(file)
#     for example in data:
#         if example["subset"] == "SQuAD":
#             SQuAD_bleu4.append(example["bleu4"])
#             total_bleu4.append(example["bleu4"])
#         if example["subset"] == "NewsQA":
#             NewsQA_bleu4.append(example["bleu4"])
#             total_bleu4.append(example["bleu4"])
#         if example["subset"] == "TriviaQA-web":
#             TriviaQA_bleu4.append(example["bleu4"])
#             total_bleu4.append(example["bleu4"])
#         if example["subset"] == "SearchQA":
#             SearchQA_bleu4.append(example["bleu4"])
#             total_bleu4.append(example["bleu4"])
#         if example["subset"] == "HotpotQA":
#             HotpotQA_bleu4.append(example["bleu4"])
#             total_bleu4.append(example["bleu4"])
#         if example["subset"] == "NaturalQuestionsShort":
#             NaturalQuestions_bleu4.append(example["bleu4"])
#             total_bleu4.append(example["bleu4"])



#     print("SQuAD_f1:",np.mean(SQuAD_f1))
#     print("NewsQA_f1:",np.mean(NewsQA_f1))
#     print("TriviaQA_f1:",np.mean(TriviaQA_f1))
#     print("SearchQA_f1:",np.mean(SearchQA_f1))
#     print("HotpotQA_f1:",np.mean(HotpotQA_f1))
#     print("NaturalQuestionsShort_f1:",np.mean(NaturalQuestions_f1))
#     print("total_f1:",np.mean(total_f1))
#     print("iid_test_samples_num:", len(total_f1))

#     print("SQuAD_bleu4:",np.mean(SQuAD_bleu4))
#     print("NewsQA_bleu4:",np.mean(NewsQA_bleu4))
#     print("TriviaQA_bleu4:",np.mean(TriviaQA_bleu4))
#     print("SearchQA_bleu4:",np.mean(SearchQA_bleu4))
#     print("HotpotQA_bleu4:",np.mean(HotpotQA_bleu4))
#     print("NaturalQuestionsShort_bleu4:",np.mean(NaturalQuestions_bleu4))
#     print("total_bleu4:",np.mean(total_bleu4))
#     print("iid_test_samples_num:", len(total_bleu4))


# iid_results = {
#     "SQuAD_f1": np.mean(SQuAD_f1).item(),
#     "NewsQA_f1": np.mean(NewsQA_f1).item(),
#     "TriviaQA_f1": np.mean(TriviaQA_f1).item(),
#     "SearchQA_f1": np.mean(SearchQA_f1).item(),
#     "HotpotQA_f1": np.mean(HotpotQA_f1).item(),
#     "NaturalQuestionsShort_f1": np.mean(NaturalQuestions_f1).item(),
#     "total_f1": np.mean(total_f1).item(),
#     "iid_test_samples_num_f1": len(total_f1),
#     "SQuAD_bleu4": np.mean(SQuAD_bleu4).item(),
#     "NewsQA_bleu4": np.mean(NewsQA_bleu4).item(),
#     "TriviaQA_bleu4": np.mean(TriviaQA_bleu4).item(),
#     "SearchQA_bleu4": np.mean(SearchQA_bleu4).item(),
#     "HotpotQA_bleu4": np.mean(HotpotQA_bleu4).item(),
#     "NaturalQuestionsShort_bleu4": np.mean(NaturalQuestions_bleu4).item(),
#     "total_bleu4": np.mean(total_bleu4).item(),
#     "iid_test_samples_num_bleu4": len(total_bleu4)
# }

# with open(output_path, 'w') as json_file:
#     json.dump(iid_results, json_file, indent=4)


import json
import sys
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--work_dir", type=str, required=True, help="Directory containing the data files")
args = parser.parse_args()

input_path = os.path.join(args.work_dir, "output", "instruction_inference_results.json")
output_path = os.path.join(args.work_dir, "output", "iid_subset_eval_results.json")

sys.setrecursionlimit(10000)

# 初始化存储F1, BLEU-4和EM分数的列表
metrics = ['rouge-f1', 'bleu4', "exact_match"]
datasets = ['SQuAD', 'NewsQA', 'TriviaQA-web', 'SearchQA', 'HotpotQA', 'NaturalQuestionsShort']
results = {dataset: {metric: [] for metric in metrics} for dataset in datasets}
total_results = {metric: [] for metric in metrics}

# 读取数据并计算各项指标
with open(input_path, 'r') as file:
    data = json.load(file)
    for example in data:
        dataset = example["subset"]
        if dataset in datasets:
            for metric in metrics:
                results[dataset][metric].append(example[metric])
                total_results[metric].append(example[metric])

# 计算平均分数并打印结果
iid_results = {}
for dataset in datasets:
    for metric in metrics:
        avg_score = np.mean(results[dataset][metric])
        print(f"{dataset}_{metric}: {avg_score}")
        iid_results[f"{dataset}_{metric}"] = avg_score.item()

for metric in metrics:
    avg_total_score = np.mean(total_results[metric])
    print(f"total_{metric}: {avg_total_score}")
    iid_results[f"total_{metric}"] = avg_total_score.item()
    iid_results[f"iid_test_samples_num_{metric}"] = len(total_results[metric])

# 写入结果到输出文件
with open(output_path, 'w') as json_file:
    json.dump(iid_results, json_file, indent=4)
