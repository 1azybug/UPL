import json
import sys
import numpy as np
from rouge import Rouge

sys.setrecursionlimit(10000)
BioASQ_f1 = []
DROP_f1 = []
DuoRC_f1 = []
RACE_f1 = []
RelationExtraction_f1 = []
TextbookQA_f1 = []
rouge = Rouge()
total = []
with open('../instruction_rank-128_lm_cl_mrqa_param_eq/instruction_inference_results.json', 'r') as file:
    data = json.load(file)
    for i in range(len(data)):
        result = 0
        if data[i]["subset"] == "BioASQ":
            scores = rouge.get_scores(data[i]["generate"], data[i]['answers'][0])
            result = max(scores[0]["rouge-1"]["f"], result)
            BioASQ_f1.append(result)
        if data[i]["subset"] == "DROP":
            scores = rouge.get_scores(data[i]["generate"], data[i]['answers'][0])
            result = max(scores[0]["rouge-1"]["f"], result)
            DROP_f1.append(result)
        if data[i]["subset"] == "DuoRC.ParaphraseRC":
            scores = rouge.get_scores(data[i]["generate"], data[i]['answers'][0])
            result = max(scores[0]["rouge-1"]["f"], result)
            DuoRC_f1.append(result)
        if data[i]["subset"] == "RACE":
            scores = rouge.get_scores(data[i]["generate"], data[i]['answers'][0])
            result = max(scores[0]["rouge-1"]["f"], result)
            RACE_f1.append(result)
        if data[i]["subset"] == "RelationExtraction":
            scores = rouge.get_scores(data[i]["generate"], data[i]['answers'][0])
            result = max(scores[0]["rouge-1"]["f"], result)
            RelationExtraction_f1.append(result)
        if data[i]["subset"] == "TextbookQA":
            scores = rouge.get_scores(data[i]["generate"], data[i]['answers'][0])
            result = max(scores[0]["rouge-1"]["f"], result)
            TextbookQA_f1.append(result)
        total.append(result)
    print("BioASQ_f1:",np.mean(BioASQ_f1))
    print("DROP_f1:",np.mean(DROP_f1))
    print("DuoRC_f1:",np.mean(DuoRC_f1))
    print("RACE_f1:",np.mean(RACE_f1))
    print("RelationExtraction_f1:",np.mean(RelationExtraction_f1))
    print("TextbookQA_f1:",np.mean(TextbookQA_f1))
    print("total:",np.mean(total))
