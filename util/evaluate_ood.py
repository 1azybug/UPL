import json
import sys
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--work_dir", type=str, required=True, help="Directory containing the data files")
args = parser.parse_args()

input_path = os.path.join(args.work_dir, "output", "instruction_inference_results.json")
output_path = os.path.join(args.work_dir, "output", "ood_subset_eval_results.json")

sys.setrecursionlimit(10000)
BioASQ_f1 = []
DROP_f1 = []
DuoRC_f1 = []
RACE_f1 = []
RelationExtraction_f1 = []
TextbookQA_f1 = []
total_f1 = []
with open(input_path, 'r') as file:
    data = json.load(file)
    for example in data:
        if example["subset"] == "BioASQ":
            BioASQ_f1.append(example["rouge-f1"])
            total_f1.append(example["rouge-f1"])
        if example["subset"] == "DROP":
            DROP_f1.append(example["rouge-f1"])
            total_f1.append(example["rouge-f1"])
        if example["subset"] == "DuoRC.ParaphraseRC":
            DuoRC_f1.append(example["rouge-f1"])
            total_f1.append(example["rouge-f1"])
        if example["subset"] == "RACE":
            RACE_f1.append(example["rouge-f1"])
            total_f1.append(example["rouge-f1"])
        if example["subset"] == "RelationExtraction":
            RelationExtraction_f1.append(example["rouge-f1"])
            total_f1.append(example["rouge-f1"])
        if example["subset"] == "TextbookQA":
            TextbookQA_f1.append(example["rouge-f1"])
            total_f1.append(example["rouge-f1"])


BioASQ_bleu4 = []
DROP_bleu4 = []
DuoRC_bleu4 = []
RACE_bleu4 = []
RelationExtraction_bleu4 = []
TextbookQA_bleu4 = []
total_bleu4 = []
with open(input_path, 'r') as file:
    data = json.load(file)
    for example in data:
        if example["subset"] == "BioASQ":
            BioASQ_bleu4.append(example["bleu4"])
            total_bleu4.append(example["bleu4"])
        if example["subset"] == "DROP":
            DROP_bleu4.append(example["bleu4"])
            total_bleu4.append(example["bleu4"])
        if example["subset"] == "DuoRC.ParaphraseRC":
            DuoRC_bleu4.append(example["bleu4"])
            total_bleu4.append(example["bleu4"])
        if example["subset"] == "RACE":
            RACE_bleu4.append(example["bleu4"])
            total_bleu4.append(example["bleu4"])
        if example["subset"] == "RelationExtraction":
            RelationExtraction_bleu4.append(example["bleu4"])
            total_bleu4.append(example["bleu4"])
        if example["subset"] == "TextbookQA":
            TextbookQA_bleu4.append(example["bleu4"])
            total_bleu4.append(example["bleu4"])



    print("BioASQ_f1:",np.mean(BioASQ_f1))
    print("DROP_f1:",np.mean(DROP_f1))
    print("DuoRC_f1:",np.mean(DuoRC_f1))
    print("RACE_f1:",np.mean(RACE_f1))
    print("RelationExtraction_f1:",np.mean(RelationExtraction_f1))
    print("TextbookQA_f1:",np.mean(TextbookQA_f1))
    print("total_f1:",np.mean(total_f1))
    print("ood_test_samples_num:", len(total_f1))

    print("BioASQ_bleu4:",np.mean(BioASQ_bleu4))
    print("DROP_bleu4:",np.mean(DROP_bleu4))
    print("DuoRC_bleu4:",np.mean(DuoRC_bleu4))
    print("RACE_bleu4:",np.mean(RACE_bleu4))
    print("RelationExtraction_bleu4:",np.mean(RelationExtraction_bleu4))
    print("TextbookQA_bleu4:",np.mean(TextbookQA_bleu4))
    print("total_bleu4:",np.mean(total_bleu4))
    print("ood_test_samples_num:", len(total_bleu4))


ood_results = {
    "BioASQ_f1": np.mean(BioASQ_f1).item(),
    "DROP_f1": np.mean(DROP_f1).item(),
    "DuoRC_f1": np.mean(DuoRC_f1).item(),
    "RACE_f1": np.mean(RACE_f1).item(),
    "RelationExtraction_f1": np.mean(RelationExtraction_f1).item(),
    "TextbookQA_f1": np.mean(TextbookQA_f1).item(),
    "total_f1": np.mean(total_f1).item(),
    "ood_test_samples_num_f1": len(total_f1),
    "BioASQ_bleu4": np.mean(BioASQ_bleu4).item(),
    "DROP_bleu4": np.mean(DROP_bleu4).item(),
    "DuoRC_bleu4": np.mean(DuoRC_bleu4).item(),
    "RACE_bleu4": np.mean(RACE_bleu4).item(),
    "RelationExtraction_bleu4": np.mean(RelationExtraction_bleu4).item(),
    "TextbookQA_bleu4": np.mean(TextbookQA_bleu4).item(),
    "total_bleu4": np.mean(total_bleu4).item(),
    "ood_test_samples_num_bleu4": len(total_bleu4)
}

with open(output_path, 'w') as json_file:
    json.dump(ood_results, json_file, indent=4)