"""Finetuning the library models for sequence classification on GLUE."""
from datasets import load_metric    
import dataclasses
import logging
import os
import errno
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, List
import torch

import numpy as np
import filelock

import transformers
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GPT2LMHeadModel
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import HfArgumentParser, TrainingArguments, set_seed
from src.gptdataset import gptDataset
from src.gpt_trainer import gptTrainer
from torch.utils.data import DataLoader

from src.dataset import FewShotDataset
from src.models import BertForPromptFinetuning, RobertaForPromptFinetuning, resize_token_type_embeddings
from src.trainer import Trainer
from src.processors import processors_mapping, num_labels_mapping, output_modes_mapping, compute_metrics_mapping, bound_mapping
from src.graft_trainer import graft_Trainer


from filelock import FileLock
from datetime import datetime

from copy import deepcopy
from tqdm import tqdm
import json
import pickle


logger = logging.getLogger(__name__)

os.environ["WANDB_DISABLED"] = "true"


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
        
    modelbase: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )    
        
        
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )

        
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    # Few-shot type
    #   - finetune: standard fine-tuning
    #   - prompt: prompt-based fine-tuning
    #   - prompt-demo: prompt-based fine-tuning with demonstrations
    few_shot_type: str = field(
        default='prompt-demo',
        metadata={"help": "Few-shot learning model type. Choice: finetune, prompt, prompt-demo"}
    )

    # Only for BERT-type model
    random_segment: bool = field(
        default=False,
        metadata={"help": "Whether to reinitialize the token type embeddings (only for BERT)."}
    )
        
    use_lm_head: int = field(
        default=1,
        metadata={"help": "0/1: Whether to use lm head or use a simple linear classifier."}
    ) 
        
        
    mask_path: str = field(
        default="",
        metadata={"help": "The path to load mask from!"}
    )
        
    sparsity_level: float = field(
        default=1e-5,
        metadata={"help": "sparsity level if initializing from highest magnitude parameters"}
    )
        
    use_CLS_linearhead: int = field(
        default=0,
        metadata={"help": "0/1: Whether to use a linear head on [CLS]"}
    )
        
    log_file_store: str = field(
        default='prompt-demo',
        metadata={"help": "File to log results"}
    )    
        
    l1_reg: float = field(
        default=0.,
        metadata={"help": "Apply l1 regularization on the model parameters!"}   
    )
    
    checkpoint_location: str = field(
        default="/tmp/best_checkpoint",
        metadata={"help": "Mask location (to save or store)?"}
    ) 
        
          

@dataclass
class DynamicDataTrainingArguments(DataTrainingArguments):
    """
    Arguments for dynamic training.
    """
    num_k: Optional[int] = field(
        default=16,
        metadata={"help": "Number of training instances per class"}
    )

    num_sample: Optional[int] = field(
        default=16,
        metadata={"help": "Number of samples (for inference) in fine-tuning with demonstrations"}
    )

    num_demo: Optional[int] = field(
        default=1,
        metadata={"help": "Number of demonstrations from each class"}
    )

    auto_demo: bool = field(
        default=True,
        metadata={"help": "Automatically generate template for using demonstrations"}
    )

    # For prompting
    template: str = field(
        default=None,
        metadata={"help": "Template"}
    )

    mapping: str = field(
        default=None,
        metadata={"help": "Label word mapping"}
    )

    template_path: str = field(
        default=None,
        metadata={"help": "Path to a txt file that stores all the templates, one per line. Do not set this when prompt_path is used"}
    )

    mapping_path: str = field(
        default=None,
        metadata={"help": "Path to a txt file that stores all the label word mappings, one per line. Do not set this when prompt_path is used"}
    )

    prompt_path: str = field(
        default=None,
        metadata={"help": "Path to a txt file that stores all the prompts (templates and mappings), one per line"}
    )
 
    template_id: int = field(
        default=None,
        metadata={"help": "Template id if using template_path"}
    )

    mapping_id: int = field(
        default=None,
        metadata={"help": "Mapping id if using template_path"}
    )

    prompt_id: int = field(
        default=None,
        metadata={"help": "Prompt id if using prompt_path"}
    )

    top_n_template: int = field(
        default=None,
        metadata={"help": "Use top-n template in the template path"}
    )

    # For logging
    tag: str = field(
        default='',
        metadata={"help": "Set the tag and find the result easier in the log."}
    )

    # For filtering when using demonstrations
    demo_filter: bool = field(
        default=False,
        metadata={"help": "Only use similar instances in demonstrations"}
    )

    demo_filter_rate: float = field(
        default=0.5,
        metadata={"help": "Only use top-x\% similar instances in demonstrations"}
    )

    demo_filter_model: str = field(
        default=None,
        metadata={"help": "Model name for demonstration filter embeddings. Will load embeddings based on the model name."}
    )

    debug_mode: bool = field(
        default=False,
        metadata={"help": "Debug mode"}
    )

    # For max length
    double_demo: bool = field(
        default=False,
        metadata={"help": "Use double length for using demonstrations"}
    )

    first_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of the first sentence (i.e., sent_0)"}
    )

    other_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of sentences other than the first sentence"}
    )

    use_full_length: bool = field(
        default=None,
        metadata={"help": "Use the full length (512)"}
    )

    # GPT-3's in-context learning (not necessary for our experiments)
    gpt3_in_context_head: bool = field(
        default=False,
        metadata={"help": "GPT-3's in-context learning (context at the beginning)"}
    )

    gpt3_in_context_tail: bool = field(
        default=False,
        metadata={"help": "GPT-3's in-context learning (context at the end)"}
    )

    gpt3_in_context_num: int = field(
        default=32,
        metadata={"help": "Number of context examples"}
    )

    truncate_head: bool = field(
        default=False,
        metadata={"help": "When exceeding the maximum length, truncate the head instead of the tail."}
    )
        
    max_length_per_example: int  = field(
        default=0,
        metadata={"help": "Max length per example."}
    )  

    # Do not set up the following fields. They are set up automatically.
    
    autoregressive: bool = field(
        default=False,
        metadata={"help": "Whether to use gpt2 fine-tuning"}
    )
        
        
    prompt: bool = field(
        default=False,
        metadata={"help": "Whether to use prompt-based fine-tuning"}
    )
    template_list: List = field(
        default=None,
        metadata={"help": "(DO NOT List of templates (only initialized after the program starts."}
    )


@dataclass
class DynamicTrainingArguments(TrainingArguments):
    # For ensemble
    array_id: int = field(
        default=-1,
        metadata={"help": "Array ID (contains seed and hyper-paramter search) to idenfity the model"}
    )

    model_id: int = field(
        default=-1,
        metadata={"help": "Model ID (contains template information) to identify the model"}
    )

    save_logit: bool = field(
        default=False,
        metadata={"help": "Save test file logit with name $TASK-$MODEL_ID-$ARRAY_ID.npy"}
    )

    save_logit_dir: str = field(
        default=None,
        metadata={"help": "Where to save the prediction result"}
    )

    # Regularization
    fix_layers: int = field(
        default=0,
        metadata={"help": "Fix bottom-n layers when optimizing"}
    )
        
    fix_embeddings: bool = field(
        default=False,
        metadata={"help": "Fix embeddings when optimizing"}
    )
    
    fix_head: bool = field(
        default=False,
        metadata={"help": "Fix lm head when optimizing"}
    )
               

    # Training
    save_at_last: bool = field(
        default=False,
        metadata={"help": "Instead of saving the best (dev performance) checkpoint, save the last checkpoint"}
    )

    # Turn off train/test
    no_train: bool = field(
        default=False,
        metadata={"help": "No training"}
    )
    no_predict: bool = field(
        default=False,
        metadata={"help": "No test"}
    )
        
       
    train_bias_only: bool = field(
        default=False,
        metadata={"help": "0/1: If we should zero out the stitch during training!"}
    )
        
    sigmoid_bias: float  = field(
        default=10.0,
        metadata={"help": "Bias inside sigmoid on the masks!"}
    )  
        
       

def main():
    parser = HfArgumentParser((ModelArguments, DynamicDataTrainingArguments, DynamicTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if 'prompt' in model_args.few_shot_type:
        data_args.prompt = True
    if 'autoregressive' in model_args.few_shot_type:
        data_args.autoregressive = True
        
        
    if training_args.no_train:
        training_args.do_train = False
    if training_args.no_predict:
        training_args.do_predict = False

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    # Load prompt/template/mapping file
    if data_args.prompt:
        if data_args.prompt_path is not None:
            assert data_args.prompt_id is not None
            prompt_list = []
            with open(data_args.prompt_path) as f:
                for line in f:
                    line = line.strip()
                    template, mapping = line.split('\t')
                    prompt_list.append((template, mapping))

            data_args.template, data_args.mapping = prompt_list[data_args.prompt_id] 
            logger.info("Specify load the %d-th prompt: %s | %s" % (data_args.prompt_id, data_args.template, data_args.mapping))
        else:
            if data_args.template_path is not None:
                with open(data_args.template_path) as f:
                    data_args.template_list = []
                    for line in f:
                        line = line.strip()
                        if len(line) > 0:
                            data_args.template_list.append(line)

                # Load top-n templates
                if data_args.top_n_template is not None:
                    data_args.template_list = data_args.template_list[:data_args.top_n_template]
                logger.info("Load top-%d templates from %s" % (len(data_args.template_list), data_args.template_path))

                # ... or load i-th template
                if data_args.template_id is not None:
                    data_args.template = data_args.template_list[data_args.template_id]
                    data_args.template_list = None
                    logger.info("Specify load the %d-th template: %s" % (data_args.template_id, data_args.template))

            if data_args.mapping_path is not None:
                assert data_args.mapping_id is not None # Only can use one label word mapping
                with open(data_args.mapping_path) as f:
                    mapping_list = []
                    for line in f:
                        line = line.strip()
                        mapping_list.append(line)

                data_args.mapping = mapping_list[data_args.mapping_id]
                logger.info("Specify using the %d-th mapping: %s" % (data_args.mapping_id, data_args.mapping))

    # Check save path
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(f"Output directory ({training_args.output_dir}) already exists.")

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = num_labels_mapping[data_args.task_name]
        output_mode = output_modes_mapping[data_args.task_name]
        logger.info("Task name: {}, number of labels: {}, output mode: {}".format(data_args.task_name, num_labels, output_mode))
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    
    special_tokens = []

    # Create tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            additional_special_tokens=special_tokens,
            cache_dir=model_args.cache_dir,
        )
    except:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.modelbase,
            additional_special_tokens=special_tokens,
            cache_dir=model_args.cache_dir,
        )

        
    data_cache_dir = data_args.data_dir
    if data_args.autoregressive:
        data_cache_dir += '_autoregressive'
    if not os.path.exists(data_cache_dir):
        os.mkdir(data_cache_dir)
    
    if data_args.autoregressive:
        dataset_class = gptDataset
    else:
        dataset_class = FewShotDataset
        
    # Get our special datasets.
    train_dataset = (
        dataset_class(data_args, tokenizer=tokenizer, cache_dir=data_cache_dir, mode="train", use_demo=("demo" in model_args.few_shot_type))
    )
    
    eval_dataset = (
        dataset_class(data_args, tokenizer=tokenizer, cache_dir=data_cache_dir, mode="dev", use_demo=("demo" in model_args.few_shot_type))
    )
    
    test_dataset = (
        dataset_class(data_args, tokenizer=tokenizer, cache_dir=data_cache_dir, mode="test", use_demo=("demo" in model_args.few_shot_type))
    )

 
    set_seed(training_args.seed)

    
    
    if not  data_args.autoregressive:
        # Create config
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
        )

    
        if 'prompt' in model_args.few_shot_type:
            if config.model_type == 'roberta':
                model_fn = RobertaForPromptFinetuning
            elif config.model_type == 'bert':
                model_fn = BertForPromptFinetuning
            else:
                raise NotImplementedError
        elif model_args.few_shot_type == 'finetune':
            model_fn = AutoModelForSequenceClassification
        else:
            raise NotImplementedError
            
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
        )
        
        def initialize_model(modelname):
        
            model = model_fn.from_pretrained(
                modelname,
                from_tf=bool(".ckpt" in modelname),
                config=config,
                cache_dir=model_args.cache_dir,
            )

            # For BERT, increase the size of the segment (token type) embeddings
            if config.model_type == 'bert':
                model.resize_token_embeddings(len(tokenizer))
                resize_token_type_embeddings(model, new_num_types=10, random_segment=model_args.random_segment)

            # Pass dataset and argument information to the model
            if data_args.prompt:
                model.label_word_list = torch.tensor(train_dataset.label_word_list).long().cuda()
            if output_modes_mapping[data_args.task_name] == 'regression':
                # lower / upper bounds
                model.lb, model.ub = bound_mapping[data_args.task_name]
            model.model_args = model_args
            model.data_args = data_args
            model.tokenizer = tokenizer
            
            if (model_args.few_shot_type == 'finetune' and model_args.use_CLS_linearhead == 1):
                model.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
                model.classifier.load_state_dict(torch.load(model_args.model_name_or_path + '/classifier'))

            return model
    else:
        config_kwargs = {}
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            **config_kwargs
        )

        if 'opt' in config.model_type:
            model_fn = OPTForCausalLM
        elif 'gpt' in config.model_type:
            model_fn = GPT2LMHeadModel
        
        
        def initialize_model(modelname):
            model = model_fn.from_pretrained(
                modelname,
                config=config,
                cache_dir=model_args.cache_dir,
            )
            return model
        
        
    
    
    model = initialize_model(model_args.modelbase)
    finetuned_model = initialize_model(model_args.model_name_or_path)
    pretrained_model = initialize_model(model_args.modelbase)
     
    if data_args.autoregressive:
        trainer_class = gptTrainer     
    else:
        trainer_class = Trainer
    
    # Initialize our Trainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    
    train_dataloader = trainer.get_train_dataloader()
    valid_dataloader = trainer.get_eval_dataloader(eval_dataset=eval_dataset)
    eval_dataloader  = trainer.get_test_dataloader(test_dataset=test_dataset)
    
    
    graft_trainer = graft_Trainer(trainer)
    
    
    model.eval()
    finetuned_model.eval()
    pretrained_model.eval()
    
    
    for param in pretrained_model.parameters():
        param.requires_grad = False
    
    for param in finetuned_model.parameters():
        param.requires_grad = False
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")        
    
    model.to(device)
    finetuned_model.to("cpu")
    pretrained_model.to("cpu")

    graft_trainer.augment_models(pretrained_model, finetuned_model, model_args, device)    
    graft_trainer.create_binary_masks()
    graft_trainer.create_basepatch()
    
    if not training_args.no_train:
        graft_trainer.train_graft(train_dataloader, \
                   valid_dataloader, \
                   eval_dataset, \
                   data_args.autoregressive, \
                   data_args.task_name, \
                  )
    
    
    #get the best trained/stored mask
    if not os.path.exists(model_args.checkpoint_location):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), model_args.checkpoint_location)
        
    graft_trainer.trainable_parameters = torch.load( model_args.checkpoint_location ) 
    final_result = {}
    graft_trainer.interpolate_model(round_=True)
    
    
    if data_args.autoregressive:
        train_score = trainer.evaluate(train_dataset).compute()
        eval_score  = trainer.evaluate(eval_dataset).compute()
        test_score  = trainer.evaluate(test_dataset).compute()
    else:
        train_score = graft_trainer.evaluate(train_dataloader, data_args.task_name)
        eval_score  = graft_trainer.evaluate(valid_dataloader, data_args.task_name)
        test_score  = graft_trainer.evaluate(eval_dataloader, data_args.task_name)
    
    final_result['train score'] = train_score.compute()
    final_result['valid score'] = eval_score.compute()
    final_result['test score']  = test_score.compute()
    
    with FileLock(model_args.log_file_store + '.lock'):
        with open(model_args.log_file_store, 'a') as f:
            final_result.update(vars(model_args))
            final_result.update(vars(training_args))
            final_result.update(vars(data_args))
            f.write(str(final_result) + '\n')
                
        
if __name__ == "__main__":
    main()
