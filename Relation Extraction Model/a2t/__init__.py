
from a2t.data import Dataset
from a2t.tasks import TopicClassificationFeatures
labels = [
    'politics', 'culture', 'economy', 'biology', 'legal', 'medicine', 'business'
]
class DummyTopicClassificationDataset(Dataset):
    def __init__(self) -> None:
        super().__init__(labels=labels)
        self.append(
            TopicClassificationFeatures(
                context="hospital: a health facility where patients"
                        " receive treatment.",
                label="medicine"
            )
        )
dataset = DummyTopicClassificationDataset()


from a2t.tasks import TopicClassificationTask
task = TopicClassificationTask(name="DummyTopic task", labels=labels)

### Defining the inference

from a2t.base import EntailmentClassifier
nlp = EntailmentClassifier(
    'roberta-large-mnli', 
    use_tqdm=False, 
    use_cuda=True, 
    half=True
)

### Putting all together

predictions = nlp(
    task=task, 
    features=dataset, 
    return_labels=True, 
    return_confidences=True, 
    topk=3
)
print(predictions)

# The result should be something close to this:
# [
#     [('medicine', 0.8545), ('biology', 0.03693), ('business', 0.0322)]
# ]

## Information Extraction with Entailment

### Defining the templates for the task
labels = ["no_relation", "per:city_of_death", "org:founded_by"]

templates = {
    "per:city_of_death": [
        "{X} died in {Y}"
    ],
    "org:founded_by": [
        "{X} was founded by {Y}",
        "{Y} founded {X}"
    ]
}

# 主要是对模型进行相对应的简化
valid_conditions = {
    "per:city_of_death": [
        "PERSON:CITY",
        "PERSON:LOCATION"
    ],
    "org:founded_by": [
        "ORGANIZATION:PERSON"
    ]
}

from a2t.tasks import BinaryTask, BinaryFeatures
task = BinaryTask(
    name="Relation Classification task",
    required_variables=["X", "Y"],
    additional_variables=["inst_type"],
    labels=labels,
    templates=templates,
    valid_conditions=valid_conditions,
    negative_label_id=0,
    multi_label=True,
    features_class=BinaryFeatures
)

### Testing the Relation Classifier

from a2t.base import EntailmentClassifier
nlp = EntailmentClassifier(
    "microsoft/deberta-v2-xlarge-mnli",
    use_tqdm=False,
    use_cuda=True, 
    half=True
)
test_examples = [
    BinaryFeatures(X='Billy Mays', Y='Tampa', inst_type='PERSON:CITY', context='Billy Mays, the bearded, boisterous pitchman who, as the undisputed king of TV yell and sell, became an unlikely pop culture icon, died at his home in Tampa, Fla, on Sunday', label='per:city_of_death'),
    BinaryFeatures(X='Old Lane Partners', Y='Pandit', inst_type='ORGANIZATION:PERSON', context='Pandit worked at the brokerage Morgan Stanley for about 11 years until 2005, when he and some Morgan Stanley colleagues quit and later founded the hedge fund Old Lane Partners.', label='org:founded_by'),
    BinaryFeatures(X='He', Y='University of Maryland in College Park', inst_type='PERSON:ORGANIZATION', context='He received an undergraduate degree from Morgan State University in 1950 and applied for admission to graduate school at the University of Maryland in College Park.', label='no_relation')
]
nlp(task=task, features=test_examples, return_labels=True, return_confidences=True)

# [('per:city_of_death', 0.98828125),
#  ('org:founded_by', 0.955078125),
#  ('no_relation', 1.0)]

__version__ = "0.3.0"

__pdoc__ = {
    "legacy": False,
    "evaluation": True,
    "tests": False,
    "tasks.base": False,
    "tasks.span_classification": False,
    "tasks.text_classification": False,
    "tasks.tuple_classification": False,
    "base.np_softmax": False,
    "base.np_sigmoid": False,
    "data.base": False,
    "data.tacred": False,
    "utils": False,
    "base.Classifier": False,
}

