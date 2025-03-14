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
with open('../experiment/example/instruction_inference_results.json', 'r') as file:
    data = json.load(file)
    for i in range(len(data)):
        result = 0
        if data[i]["subset"] == "SQuAD":
            scores = rouge.get_scores(data[i]["generate"], data[i]['answers'][0])
            result = max(scores[0]["rouge-1"]["f"], result)
            BioASQ_f1.append(result)
        if data[i]["subset"] == "NewsQA":
            scores = rouge.get_scores(data[i]["generate"], data[i]['answers'][0])
            result = max(scores[0]["rouge-1"]["f"], result)
            DROP_f1.append(result)
        if data[i]["subset"] == "TriviaQA-web":
            scores = rouge.get_scores(data[i]["generate"], data[i]['answers'][0])
            result = max(scores[0]["rouge-1"]["f"], result)
            DuoRC_f1.append(result)
        if data[i]["subset"] == "SearchQA":
            scores = rouge.get_scores(data[i]["generate"], data[i]['answers'][0])
            result = max(scores[0]["rouge-1"]["f"], result)
            RACE_f1.append(result)
        if data[i]["subset"] == "HotpotQA":
            scores = rouge.get_scores(data[i]["generate"], data[i]['answers'][0])
            result = max(scores[0]["rouge-1"]["f"], result)
            RelationExtraction_f1.append(result)
        if data[i]["subset"] == "NaturalQuestionsShort":
            scores = rouge.get_scores(data[i]["generate"], data[i]['answers'][0])
            result = max(scores[0]["rouge-1"]["f"], result)
            TextbookQA_f1.append(result)
        total.append(result)
    print("SQuAD_f1:",np.mean(BioASQ_f1))
    print("NewsQA_f1:",np.mean(DROP_f1))
    print("TriviaQA_f1:",np.mean(DuoRC_f1))
    print("SearchQA_f1:",np.mean(RACE_f1))
    print("HotpotQA_f1:",np.mean(RelationExtraction_f1))
    print("NaturalQuestionsShort_f1:",np.mean(TextbookQA_f1))
    print("total:",np.mean(total))
