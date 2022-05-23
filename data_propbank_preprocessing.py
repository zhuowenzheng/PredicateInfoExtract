import json
import os

from allennlp.predictors import Predictor


def get_predictor():
    return Predictor.from_path("/Users/alexzheng/Downloads/structured-prediction-srl-bert.2020.12.15.tar.gz")


print("---------")
print(os.getcwd())
print("---------")

raw_data_path = "/Users/alexzheng/Documents/textual_entailment/PredicateInfoExtract/preprocessed/openbookqa/openbookqa-test-processed-questions.jsonl"

preprocessed_data_path = "/Users/alexzheng/Documents/textual_entailment/PredicateInfoExtract/preprocessed/openbookqa/srl-processed-premises.jsonl"

target_dir = os.path.dirname(preprocessed_data_path)

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

predictor = get_predictor()

with open(preprocessed_data_path, "w") as write_file:
    # 1.load raw data
    with open(raw_data_path, 'r') as entailment_file:

        # print("enter 3")
        for line in entailment_file:

            if line.strip():
                instances_json = json.loads(line.strip())
                premises = instances_json["premises"]


            # 2.process each premise
            count = 0
            premisesReplace = []

            for premise in premises:
                count += 1
                premiseReplace = ""

                print()
                print("------------------------------")
                print('Number of processed premise: ', count)
                print('premise:', premise)
                print()
                print('prediction:\n', predictor.predict(premise))
                try:
                    premiseReplace = predictor.predict(premise)['verbs'][0]['description']
                except:
                    premiseReplace = premise
                    print("Same as original.Thrown.")
                else:
                    print('result:\n', premiseReplace)
                    print("------------------------------")
                    premisesReplace.append(premiseReplace)

            instance = {}
            instance["premises"] = premisesReplace

            write_file.write(json.dumps(instance) + "\n")
