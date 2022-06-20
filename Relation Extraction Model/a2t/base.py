"""The module `base` implements all the basic methods to perform the inference, including the `EntailmentClassifier`.
"""
import os
import sys
import gc
from typing import List

import numpy as np
import torch

# all the basic methods to perform the inference
# 使用tqdm(是一种python的内存条:主要是方便显示内容进度的)
try:
    from tqdm import tqdm

    _use_tqdm = True
except ImportError:
    _use_tqdm = False

# transformers(直接引入相关的包进行使用)
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

# change2 .
from a2t.tasks import Features, Task

try:
    import transformers
    # 没有transformer相关的包就报错(这个可能是老版本的问题了)
    # transformers.logging.set_verbosity_error()
except ImportError:
    pass

# softmax
def np_softmax(x, dim=-1):
    e = np.exp(x)
    return e / np.sum(e, axis=dim, keepdims=True)

# sigmoid
def np_sigmoid(x, dim=-1):
    return 1 / (1 + np.exp(-x))


class Classifier(object):
    """Abstact classifier class."""

    def __init__(
        self, labels: List[str], pretrained_model: str = "roberta-large-mnli", use_cuda=True, half=False, verbose=True
    ):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        self.labels = labels
        self.use_cuda = use_cuda
        self.half = half
        # verbose:输出日志
        self.verbose = verbose

        # Supress stdout printing for model downloads
        if not verbose:
            # 没有日志就自己打开一个写进去 + 加载预模型
            sys.stdout = open(os.devnull, "w")
            self._initialize(pretrained_model)
            sys.stdout = sys.__stdout__
        else:
            # 有了日志就直接加载模型就好
            self._initialize(pretrained_model)

        self.model = self.model.to(self.device)
        self.model = self.model.eval()
        # model.eval() is a kind of switch for some specific layers/parts of the model that behave
        # differently during training and inference (evaluating) time. For example, Dropouts Layers,
        # BatchNorm Layers etc. You need to turn off them during model evaluation, and .eval() will
        # do it for you. In addition, the common practice for evaluating/validation is using
        # torch.no_grad() in pair with model.eval() to turn off gradients computation:
        if self.use_cuda and self.half and torch.cuda.is_available():
            self.model = self.model.half()

    def _initialize(self, pretrained_model):
        raise NotImplementedError

    def __call__(self, context, batch_size=1):
        raise NotImplementedError

    def clear_gpu_memory(self):
        self.model.cpu()
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


class EntailmentClassifier(Classifier):
    """General purpose Entailment based classifier.

    This class contains the code for entailment-based zero-shot classification inference. It is pretended to be
    task and data independent.
    """

    def __init__(
        self,
        pretrained_model: str = "roberta-large-mnli",
        use_cuda: bool = True,
        # use half precision if possible
        half: bool = False,
        # output log information
        verbose: bool = True,
        # tqdm是一个小工具,显示相关进度条的
        use_tqdm: bool = True,
        **kwargs
    ):
        """
        Args:
            pretrained_model (str, optional): The name or path of the pretrained model. Defaults to "roberta-large-mnli".
            use_cuda (bool, optional): Use the GPU if possible. Defaults to True.
            half (bool, optional): Use half precision if possible. Defaults to False.
            verbose (bool, optional): Output log information. Defaults to True.
        """
        super().__init__(None, pretrained_model, use_cuda, half, verbose)
        self.use_tqdm = use_tqdm and _use_tqdm

    def _initialize(self, pretrained_model: str):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model)
        self.config = AutoConfig.from_pretrained(pretrained_model)
        self.ent_pos = self.config.label2id.get("ENTAILMENT", self.config.label2id.get("entailment", None))
        if self.ent_pos is None:
            raise ValueError("The model config must contain ENTAILMENT label in the label2id dict.")
        else:
            self.ent_pos = int(self.ent_pos)

    # 设置门槛(主要是抛弃掉一些概率比较小的组合)
    @staticmethod
    def apply_threshold(
        output: np.ndarray,
        threshold: float = 0.0,
        ignore_negative_prediction: bool = True,
        application_type: str = "prediction",
    ) -> np.ndarray:
        """
        Args:
            output (ndarray): (batch_size, n_labels) The predicted probabilities.
            threshold (float): The threshold value to apply.
            ignore_negative_prediction (bool): Ignore the negative prediction probabilites. Default to True.
            application_type (str): How to apply the threshold: Options:
                * **"prediction"**: Set to 1.0 the probability of the negative class if the no prediction is higher than the threshold.
                * **"mask"**: Set to 0.0 the probabilities of the positive classes that are lower or equal to the threshold.
        """
        output_ = output.copy()
        # Ignore the negative prediction probabilities
        # 具体要想清楚
        if ignore_negative_prediction:
            output_[:, 0] = 0.0
        if application_type == "prediction":
            activations = (output_ >= threshold).sum(-1).astype(int)
            output_[activations == 0, 0] = 1.00
        elif application_type == "mask":
            activations = output_ < threshold
            output_[activations] = 0.0
        else:
            raise ValueError("""application_type argument must be "prediction" or "mask".""")

        return output_

    def __call__(
        self,
        task: Task,
        features: List[Features],
        negative_threshold: float = 0.5,
        # Return the first `k` predictions with higher probabilities. Defaults to 1
        topk: int = 1,
        return_labels: bool = False,
        return_confidences: bool = False,
        ignore_negative_prediction: bool = False,
        return_raw_output: bool = False,
        **kwargs
    ) -> List:
        """Call method for the EntailmentClassifier.


        TODO: Add output documentation.

        Args:
            task (Task): The task instance used for inference.
            features (List[Features]): The list of features to classify.
            negative_threshold (float, optional): The threshold used if necessary. Defaults to 0.5.
            topk (int, optional): Return the first `k` predictions with higher probabilities. Defaults to 1.
            return_labels (bool, optional): Whether to return the label ids or names. Defaults to False (ids).
            return_confidences (bool, optional): Whether to return prediction confidences or not. Defaults to False.
            ignore_negative_prediction (bool, optional): Whether to ignore the predictions of the negative class. Defaults to False.
            return_raw_output (bool, optional): Return the raw output along with the processed one. Defaults to False.

        Returns:
            List: A list with the predictions.
        """
        task.assert_features_class(features)

        # return confidence

        outputs = []
        iterator = features if not self.use_tqdm else tqdm(features, total=len(features))
        # 具体再看看
        with torch.no_grad():
            for feature in iterator:

                # 1.feature 里面 进行 generation
                sentence_pairs = task.generate_premise_hypotheses_pairs([feature], self.tokenizer.sep_token)

                print("---sentence pairs---")
                print(sentence_pairs)

                # 2.给所有的数据 加上 相对应的ID符号
                data = self.tokenizer(sentence_pairs, return_tensors="pt", padding=True).input_ids
                data = data.to(self.device)
                output = self.model(data)[0].detach().cpu().numpy()
                outputs.append(output)

        outputs = np.vstack(outputs)

        # 判断 (这个地方稍微要讨论下)
        if task.multi_label:
            outputs = np.exp(outputs) / np.exp(outputs).sum(-1, keepdims=True)
        outputs = outputs[..., self.ent_pos].reshape(len(features), -1)

        # 判断前面一部分的值
        preds = task.reverse_to_labels(outputs)

        if not task.multi_label:
            preds = np_softmax(preds)

        preds = task.apply_valid_conditions(features, preds)

        # 用之前设定的标准来淘汰一些值
        apply_threshold = task.multi_label and negative_threshold > 0
        if apply_threshold:
            preds = self.apply_threshold(
                preds, threshold=negative_threshold, ignore_negative_prediction=ignore_negative_prediction
            )

        predictions = np.argsort(preds, -1)[:, ::-1]


        if topk > 0:
            predictions = predictions[:, :topk]
        if return_labels:
            predictions = task.idx2label(predictions)
        if return_confidences:
            confidences = np.sort(preds, -1)[:, ::-1]
            if topk > 0:
                confidences = confidences[:, :topk]

            predictions = np.stack((predictions, confidences), -1).tolist()
            predictions = [
                [(int(label), float(conf)) if not return_labels else (label, float(conf)) for label, conf in row]
                for row in predictions
            ]
        else:
            predictions = predictions.tolist()



        if topk == 1:
            predictions = [row[0] for row in predictions]



        if return_raw_output:
            return (predictions, preds)
        else:
            return predictions


__pdoc__ = {"EntailmentClassifier.__call__": True}
