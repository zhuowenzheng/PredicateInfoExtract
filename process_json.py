import numpy as np
import stanza
import json
import os
from collections import defaultdict

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader


def cutstr(str):

    final = []
    result = str.split(',')

    for i in result:
        m = i.split('.')
        final += m

    return final

print("---------")
print(os.getcwd())
print("---------")

stanza.download('en')
nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')

raw_data_path = cached_path("openbookqa-test-processed-questions.jsonl")

preprocessed_data_path = f"/openbookqa-test-hypothesis-processed-questions.jsonl"

target_dir = os.path.dirname(preprocessed_data_path)

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

file_path = cached_path("openbookqa-test-processed-questions.jsonl")
with open(preprocessed_data_path, "w") as write_file:

    with open(raw_data_path, 'r') as entailment_file:

        for line in entailment_file:

            if line.strip():

                instances_json = json.loads(line.strip())
                premises = instances_json["premises"]
                raw_question = instances_json["raw_question"]
                question_id = instances_json["question_id"]
                hypotheses = instances_json["hypotheses"]
                entailments = instances_json.get("entailments", None)

            hypotheses_cut = []
            entity_types = []

            for hypothesis in hypotheses:

                hypothesis_cut = []
                entity_type = []

                temp = cutstr(hypothesis)

                for sentence in temp:

                    doc = nlp(sentence)

                    for sent in doc.sentences:

                        if len(sent.ents) == 0:
                            hypothesis_cut.append(sent)
                            entity_type.append(defaultdict())


                        else:
                            hypothesis_cut.append(sent)
                            # hypothesisReplace += sentence.replace(ent.text,"<" + ent.type + ">" + ent.text + "</" + ent.type + ">")
                            entities = dict(zip([ent.text for ent in sent.ents],[ent.type for ent in sent.ents]))
                            entity_type.append(entities)


                hypotheses_cut.append(hypothesis_cut)
                entity_types.append(entity_type)


            print(hypotheses)
            print(entity_types)
            instance = {}
            instance["premises"] = premises
            instance["raw_question"] = raw_question
            instance["question_id"] = question_id
            instance["hypotheses"] = hypotheses_cut
            instance["entity_relations"] = entity_types
            instance["entailments"] = entailments

            write_file.write(json.dumps(instance)+"\n")





