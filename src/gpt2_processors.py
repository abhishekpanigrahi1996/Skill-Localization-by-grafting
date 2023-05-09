"""Dataset utils for different data settings for GLUE."""

import os
import logging
from transformers import DataProcessor, InputExample
from transformers.data.processors.glue import *
from transformers.data.metrics import glue_compute_metrics
import pandas as pd
import logging
import json


logger = logging.getLogger(__name__)

class Mrpc_gptProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = f" {line[3]}\n question: {line[4]} true or false?\n answer:" 
            #f" question: Given that \"{line[3]}\" Is \"{line[4]}\" true, or false?\n answer:" 
            text_b = [' false', ' true' ] 
            label = text_b[int(line[0])]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
    
    
    
class QQP_gptProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question1"].numpy().decode("utf-8"),
            tensor_dict["question2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]
    
    
    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = f" {line[3]}\n question: {line[4]} true or false?\n answer:" 
            #f" question: Given that \"{line[3]}\" Is \"{line[4]}\" true, or false?\n answer:" 
            text_b = [' false', ' true' ] 
            label = text_b[int(line[5])]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
    
    
    
    
class Mnli_gptProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        
        label_map = {}
        label_map["contradiction"] = 0
        label_map["entailment"] = 1
        label_map["neutral"] = 2
        
        
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = f" {line[8]}\n question: {line[9]} true, false, or neither?\n answer:" 
            #f" question: Given that \"{line[8]}\" Is \"{line[9]}\" true, false, or neither?\n answer:" 
            text_b = [' false', ' true', ' neither'] 
            label = text_b[label_map[line[-1]]]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
        


class Snli_gptProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        label_map = {}
        label_map["contradiction"] = 0
        label_map["entailment"] = 1
        label_map["neutral"] = 2
        
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = f" {line[7]}\n question: {line[8]} true, false, or neither?\n answer:" 
            #f" question: Given that \"{line[7]}\" Is \"{line[8]}\" true, false, or neither?\n answer:" 
            text_b = [' false', ' true', ' neither'] 
            label = text_b[label_map[line[-1]]]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
    
class Rte_gptProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        label_map = {}
        label_map["not_entailment"] = 0
        label_map["entailment"] = 1
        
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = f" {line[1]}\n question: {line[2]} true or false?\n answer:" 
            #f" {line[1]}\n question: {line[2]} true or false?\n answer:"
            text_b = [' false', ' true'] 
            label = text_b[label_map[line[-1]]]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class Qnli_gptProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        label_map = {}
        label_map["not_entailment"] = 0
        label_map["entailment"] = 1
        
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = f" {line[1]}\n question: {line[2]} true or false?\n answer:"  
            #f" {line[1]}\n question: {line[2]} true or false?\n answer:"
            text_b = [' false', ' true'] 
            label = text_b[label_map[line[-1]]]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
    
class Sst2_gptProcessor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 0
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = f"\"{line[text_index]}\" has a tone that is"
            text_b = [' negative', ' positive']
            label  = text_b[int(line[1])]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    
class mr_cr_gptProcessor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "train.csv"), header=None).values.tolist(), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.csv"), header=None).values.tolist(), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "test.csv"), header=None).values.tolist(), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 1
        for (i, line) in enumerate(lines):
            #if i == 0:
            #    continue
            guid = "%s-%s" % (set_type, i)
            text_a = f"\"{line[text_index]}\" has a tone that is"
            text_b = [' negative', ' positive']
            label  = text_b[int(line[0])]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class subj_gptProcessor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "train.csv"), header=None).values.tolist(), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.csv"), header=None).values.tolist(), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "test.csv"), header=None).values.tolist(), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 1
        for (i, line) in enumerate(lines):
            #if i == 0:
            #    continue
            guid = "%s-%s" % (set_type, i)
            text_a = f"\"{line[text_index]}\" has a tone that is"
            text_b = [' object', ' subject']
            label  = text_b[int(line[0])]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
    
    
class trec_gptProcessor(DataProcessor):
    """Processor for the trec data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )


    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "train.csv"), header=None).values.tolist(), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.csv"), header=None).values.tolist(), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "test.csv"), header=None).values.tolist(), "test")
    
    
    def get_labels(self):
        """See base class."""   
        return list(range(6))

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 1
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = f' {line[text_index]} The answer to this question will be'
            text_b = [" a description.", " an entity.", " a location.", " a number.", " an abbreviation.", " a person."]
            label  = text_b[int(line[0])]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class agnews_gptProcessor(DataProcessor):
    """Processor for the trec data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )


    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")
    
    def get_labels(self):
        """See base class."""   
        return list(range(4))

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 0
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = f' summary: {line[text_index]}\n topic:'
            text_b = [ ' World', ' Sports', ' Business', ' Science' ]
            label  = text_b[int(line[1])]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    
class gptProcessor_noprompt(DataProcessor):
    """
    Data processor for text classification datasets (mr, sst-5, subj, trec, cr, mpqa).
    """
    def __init__(self):
        self.labels = None
    
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )
    
    def load_data(self, data_path, is_null=False):
        data = []
        data_path = data_path#os.path.join(data_dir, "{}.jsonl".format(split))
        with open(data_path, "r") as f:
            for line in f:
                dp = json.loads(line)
                if is_null:
                    dp["input"] = "N/A"
                data.append(dp)
        #if "options" in data[0]:
        #    self.labels = ["\n" + label for label in data[0]["options"]]
        return data


    def get_train_examples(self, data_path):
        """See base class."""
        return self._create_examples(self.load_data(data_path + "train.jsonl"), "train")

    def get_dev_examples(self, data_path):
        """See base class."""
        return self._create_examples(self.load_data(data_path + "dev.jsonl"), "dev")

    def get_test_examples(self, data_path):
        """See base class."""
        return self._create_examples(self.load_data(data_path + "test.jsonl"), "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            examples.append(InputExample(guid=i, text_a=line["input"], text_b=["\n" + opt for opt in line["options"]], label="\n" + line["output"]))
        return examples
    
    
    
def text_classification_metrics(task_name, preds, labels):
    return {"acc": (preds == labels).mean()}

# Add your task to the following mappings
processors_mapping_gpt = {
    "sst-2": Sst2_gptProcessor(),
    "mr": mr_cr_gptProcessor(),
    "cr": mr_cr_gptProcessor(),
    "mpqa": mr_cr_gptProcessor(),
    "subj": subj_gptProcessor(),
    "snli": Snli_gptProcessor(),
    "mnli": Mnli_gptProcessor(),
    "qnli": Qnli_gptProcessor(),
    "qqp": QQP_gptProcessor(),
    "mrpc": Mrpc_gptProcessor(),
    "ag_news": agnews_gptProcessor(),
    "trec": trec_gptProcessor(),
    "rte": Rte_gptProcessor(),
    "no_prompt": gptProcessor_noprompt(),
}

empty_template_mapping = {
    "sst-2": " The quote has a tone that is", 
    "mr": " The quote has a tone that is", 
    "cr": " The quote has a tone that is", 
    "mpqa": " The quote has a tone that is", 
    "subj": " The quote has a tone that is", 
    "snli": " true, false, or neither?\n answer:",
    "mnli": " true, false, or neither?\n answer:",
    "qnli": " true or false?\n answer:",
    "qqp": " true or false?\n answer:",
    "mrpc": " true or false?\n answer:",
    "trec": " The answer to this question will be",
    "rte": " true or false?\n answer:",
    "ag_news": " \n topic:",
    "no_prompt": "N/A",
}    
    

#not necessary for gpt experiments
num_labels_mapping = {}
output_modes_mapping = {}
# Return a function that takes (task_name, preds, labels) as inputs
compute_metrics_mapping = {}

# For regression task only: median
median_mapping = {}
bound_mapping = {}