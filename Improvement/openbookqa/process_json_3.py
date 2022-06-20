import numpy as np
# import stanza
import json
import os

print(os.getcwd())

raw_data_path = "./data_processed/openbookqa-wrong-processed-questions.jsonl"
label = 0
with open(raw_data_path, 'r') as entailment_file:
    for line in entailment_file:

        # print(type(line))
        print(line)

        label += 1
        if label == 5:
            break




