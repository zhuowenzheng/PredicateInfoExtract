import numpy as np
import stanza
import json
import os

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

def cutstr(str):

    final = []
    result = str.split(',')

    for i in result:
        m = i.split('.')
        final += m

    return final

# print("---------")
# print(os.getcwd())
# print("---------")

stanza.download('en')
nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')

raw_data_path = "data/preprocessed/openbookqa/openbookqa-test-processed-questions.jsonl"

print("label 1")

preprocessed_data_path = "data/preprocessed/openbookqa/openbookqa-test-hypothesis-processed-questions.jsonl"

print("laebl 2")

target_dir = os.path.dirname(preprocessed_data_path)

if not os.path.exists(target_dir):
    os.makedirs(target_dir)


print("enter 1")

with open(preprocessed_data_path, "w") as write_file:

    print("enter 2")
    # 1.处理相关的部分 (将文件读入 + 将hypotheses每个部分输出出来)
    with open(raw_data_path, 'r') as entailment_file:

    # print("enter 3")
        for line in entailment_file:

            if line.strip():

                instances_json = json.loads(line.strip())
                premises = instances_json["premises"]
                raw_question = instances_json["raw_question"]
                question_id = instances_json["question_id"]
                hypotheses = instances_json["hypotheses"]
                entailments = instances_json.get("entailments", None)

            # 2.将hypotheses做相应的处理
            # count = 0
            hypothesesReplace = []

            for hypothese in hypotheses:

                hypotheseReplace = ""
                temp = cutstr(hypothese)
                tempartnumber = len(temp)
                count = 0

                for sentence in temp:

                    doc = nlp(sentence)

                    for sent in doc.sentences:

                        if len(sent.ents) == 0:
                            print("---Here---")
                            hypotheseReplace += sentence
                        else:
                            for ent in sent.ents:
                                hypotheseReplace += sentence.replace(ent.text,"<" + ent.type + ">" + ent.text + "<" + ent.type + ">")

                        if tempartnumber == 1:
                            pass
                        if count < tempartnumber - 1:
                            hypotheseReplace += " , "
                            count += 1

                hypothesesReplace.append(hypotheseReplace)

            print(hypotheses)
            print(hypothesesReplace)



                #doc = nlp(hypothesis)
            #     count+=1
            #
            # if count == 2:
            #     break

            instance = {}
            instance["premises"] = premises
            instance["raw_question"] = raw_question
            instance["question_id"] = question_id
            instance["hypotheses"] = hypothesesReplace
            instance["entailments"] = entailments

            write_file.write(json.dumps(instance)+"\n")



