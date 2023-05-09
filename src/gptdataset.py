"""Dataset utils for different data settings for GLUE."""

import os
import logging
import torch
import numpy as np
import time
from filelock import FileLock
import json
from src.gpt2_processors import processors_mapping_gpt, num_labels_mapping, output_modes_mapping, compute_metrics_mapping, median_mapping, empty_template_mapping
import dataclasses
from src.dataset import OurInputFeatures
import random


from dataclasses import dataclass
from typing import List, Optional, Union
import pandas as pd
from transformers.data.processors.utils import InputFeatures

logger = logging.getLogger(__name__)


class gptDataset(torch.utils.data.Dataset):
    """Dataset for in-context learning."""
    def __init__(self, args, tokenizer, cache_dir=None, mode="train", use_demo=False, is_null=False):
        self.args = args
        self.task_name = args.task_name
        
        #if self.args.gpt_prompt:
        self.processor = processors_mapping_gpt[self.task_name]
        self.empty_template = empty_template_mapping[self.task_name]
        #else:
        #    self.processor = processors_mapping_gpt["no_prompt"]
        #    self.empty_template =  empty_template_mapping["no_prompt"]
        self.tokenizer = tokenizer
        self.use_demonstrations = use_demo
        self.mode = mode
        self.is_null = is_null
        
        
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}".format(
                mode,
                tokenizer.__class__.__name__,
                str(args.max_seq_length),
                args.task_name,
            ),
        )

        logger.info(f"Creating/loading examples from dataset file at {args.data_dir}")

        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.support_examples, self.query_examples = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")

                # The support examples are sourced from the training set.
                self.support_examples = self.processor.get_train_examples(args.data_dir)

                if mode == "dev":
                    self.query_examples = self.processor.get_dev_examples(args.data_dir)
                elif mode == "test":
                    self.query_examples = self.processor.get_test_examples(args.data_dir)
                else:
                    self.query_examples = self.support_examples

                start = time.time()
                torch.save([self.support_examples, self.query_examples], cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )
        

        self.num_labels = len(self.query_examples[0].text_b)
        if self.use_demonstrations:
            self.num_k = len(self.support_examples) 
        else:
            self.num_k = 0 
        self.create_features(self.query_examples, self.support_examples, verbose=(self.mode=='train'))
        self.size = len(self.features)
         

    def __getitem__(self, i):
        features = self.features[i]
        return features
    
    
    def enc(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)
    
    
    def __len__(self):
        return self.size
    
    
    def preprocess_single_example(self, input_, for_demonstration, max_length, use_separator):
        if use_separator:
            separator  = self.enc("\n\n\n")
        else:
            separator  = self.enc("")
            
            
        input_text = input_.text_a
        options    = input_.text_b
        label      = input_.label
        
        input_tokens = self.enc(input_text)
        output_tokens = self.enc(label)
            
        if for_demonstration:
            if self.args.truncate_head:
                if (self.task_name.startswith("inst:piqa") or self.task_name.startswith("inst:yahoo_answers_topics")) and \
                            len(separator) + len(input_tokens)+len(output_tokens)+2>max_length:
                    input_tokens = input_tokens[-(max_length // 2):]
                    output_tokens = output_tokens[-(max_length // 2 - 2 - len(separator) ):]

                elif len(input_tokens)>=max_length - 2 - len(output_tokens) - len(separator):
                    if self.task_name.startswith("inst:") and len(input_tokens)<len(output_tokens):
                        output_tokens = output_tokens[-(max_length - 2 - len(input_tokens) - len(separator)):]
                    else:
                        input_tokens = input_tokens[-(max_length - 2 - len(output_tokens)- len(separator) ):]
                  
            else:    
                if (self.task_name.startswith("inst:piqa") or self.task_name.startswith("inst:yahoo_answers_topics")) and \
                            len(separator) + len(input_tokens)+len(output_tokens)+2>max_length:
                    input_tokens = input_tokens[:max_length // 2]
                    output_tokens = output_tokens[:max_length // 2 - 2 - len(separator)]

                elif len(input_tokens)>=max_length - 2 - len(output_tokens) - len(separator) :
                    if self.task_name.startswith("inst:") and len(input_tokens)<len(output_tokens):
                        output_tokens = output_tokens[:max_length - 2 - len(input_tokens) - len(separator)]
                    else:
                        input_tokens = input_tokens[:max_length - 2 - len(output_tokens) - len(separator)]
            input_tokens =   separator  + input_tokens                      
            query_type_ids = [1 for _ in input_tokens] + [0 for _ in output_tokens] 

            return input_tokens, output_tokens, query_type_ids         
        else:
            
            #if self.mode != 'train':
            all_option_tokens = [ self.enc(opt) for opt in options ]
            #else:
            option_tokens = [ self.enc(label) ]
            all_option_length = [len(option) for option in option_tokens]
            option_length = np.max(all_option_length)

            if len(input_tokens)>=max_length - 2 - option_length - len(separator):
                if self.args.truncate_head:
                    input_tokens = input_tokens[-(max_length - 2 - option_length - len(separator) ):]
                else:
                    input_tokens = input_tokens[:max_length - 2 - option_length - len(separator)]
            input_tokens =   separator  + input_tokens          
            query_type_ids = [ [1 for _ in input_tokens]  for _ in option_tokens ] 
            
            
      
            input_tokens = [input_tokens for _ in option_tokens]
            output_tokens = option_tokens
            answer = options.index(label)
            
            
            
            return input_tokens, output_tokens, answer, query_type_ids, all_option_tokens 
            
    def preprocess_pair_example(self, input_, query_ids_, option_, max_length):
        
       
        if self.args.truncate_head and len(input_)+len(option_) > max_length:
            input_ = input_[len(input_)+len(option_)-max_length:]            
            query_ids_ = query_ids_[len(query_ids_)+len(option_)-max_length:]
            
            
            
        n_mask = max_length-len(input_)-len(option_)
        query_ids_ += [0 for _ in option_] + [0 for _ in range(n_mask)]
        
        input_ids = input_+option_+[0 for _ in range(n_mask)]
        attention_mask = [1 for _ in input_+option_] + [0 for _ in range(n_mask)]
        token_type_ids = [0 for _ in input_] + [1 for _ in option_] + [0 for _ in range(n_mask)]
        
        assert len(query_ids_) == max_length
        assert len(input_ids) == max_length
        assert len(attention_mask) == max_length
        assert len(token_type_ids) == max_length
        
        return input_ids, attention_mask, token_type_ids, query_ids_
 
        
    
    
    def create_features(
        self,
        test_examples,
        supports,
        verbose=False
    ):
        """
        Returns a list of processed "InputFeatures".
        """
        max_length = min(self.args.max_length_per_example * (self.args.num_k + 1), self.args.max_seq_length)
        max_length_per_example = self.args.max_length_per_example

        demonstrations = []
        query_ids = []
 

        
        
        if self.use_demonstrations:
            for support in supports:
                support_input, support_label, query_type_ids  = self.preprocess_single_example(support, for_demonstration=True, max_length=max_length_per_example, use_separator=self.use_demonstrations)
                demonstrations += support_input + support_label  
                query_ids += query_type_ids

            

        self.correct_answers = []
        self.features = []
        self.indices  = []
        for index, test_data in enumerate(test_examples):
            if self.is_null:
                test_data.text_a = self.empty_template
           
            test_inputs, test_options, test_answer, query_type_ids, all_options = self.preprocess_single_example(test_data, for_demonstration=False, max_length=max_length_per_example, use_separator=self.use_demonstrations)
            
             
            
            self.correct_answers += [test_answer]
            self.indices += [ range( len(self.features) , len(self.features) + len(test_inputs)  ) ] 
            
            counter = 0
            for test_input_, test_option_ in zip(test_inputs, test_options) :
                result = {}
                input_ = demonstrations + test_input_
                query_ids_ = query_ids  + query_type_ids[counter]
                counter += 1
                
                augment = test_option_
                input_ids, attention_mask, token_type_ids, query_ids_ = self.preprocess_pair_example(input_, query_ids_,  augment, max_length)
                
                result['input_ids'], result['attention_mask'], result['token_type_ids'], result['label_word_list'], result['label'] = input_ids, attention_mask, token_type_ids,  [opt[0] for opt in all_options], test_answer
                
               
                self.features.append( OurInputFeatures(**result) )

                if verbose and index == 0:
                    logger.info("*** Example ***")
                    logger.info("features: %s" % result)
                    
                    logger.info("text: %s" % self.tokenizer.decode(result['input_ids']))
       
       
        