import torch
import transformers
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import math 
from transformers import get_constant_schedule_with_warmup, get_constant_schedule, get_linear_schedule_with_warmup
from datasets import load_metric    

class gptTrainer(transformers.Trainer):
    
    def create_optimizer_and_scheduler(self, num_training_steps: Optional[int] = 0):
        """
        Based on Transformers' default one, we add fixing layer option where the bottom n layers' parameters
        are fixed and only the top layers are further fine-tuned.
        """
        self.num_training_steps = num_training_steps
        #if self.optimizer is  None:
        self.select_trainable_parameters()
        self.no_decay = ["bias", "LayerNorm.weight"]
        self.init_opt(self.args.weight_decay, self.args.learning_rate)
            
        
    def select_trainable_parameters(self):
        self.params = {}
        for n, p in self.model.named_parameters():

            if self.args.fix_embeddings and ('wte' in n or 'wpe' in n):
                print('no ', n)  
            elif ('wte' in n or 'wpe' in n):
                print('yes', n)
                self.params [n] = p

            if self.args.fix_head and 'ln_f' in n:
                print('no ', n)            
            elif 'ln_f' in n:
                print('yes', n)
                self.params [n] = p


            if self.args.fix_layers > 0:   
                if 'transformer.h.' in n:
                    try:
                        layer_num = int(n[n.find('transformer.h.') + len('transformer.h.'):].split('.')[0])

                    except:
                        print(n)
                        raise Exception("")
                    if layer_num >= self.args.fix_layers:
                        print('yes', n)
                        self.params[n] = p
                    else:
                        print('no ', n)
            else:
                if 'transformer.h.' in n:
                    print('yes', n)
                    self.params[n] = p

        

    def init_opt(self, weight_decay, learning_rate):
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.params.items() if not any(nd in n for nd in self.no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.params.items() if any(nd in n for nd in self.no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.args.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
            if self.lr_scheduler is None:
                self.lr_scheduler = get_linear_schedule_with_warmup(
                    self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.num_training_steps
                )


        elif self.args.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(
                optimizer_grouped_parameters,
                lr=learning_rate
            )
            if self.lr_scheduler is None:
                self.lr_scheduler = get_constant_schedule(
                    self.optimizer
                )     
        else:
            raise NotImplementedError
        
        
        
    def finetune(self, train_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        
        
        train_dataloader = self.get_train_dataloader()
        
        
        self.model.train()
        
        self.global_step = 0
        self.objective = 0.
        counter = 0
        self.optimizer.zero_grad()
        
        pbar = tqdm(total=self.args.max_steps)

        while(1):
            
            if self.global_step >= self.args.max_steps :
                break

            for batch in train_dataloader:
                
                pbar.update(1)
                if self.global_step >= self.args.max_steps :
                    break
                
                
                
                input_ids=batch["input_ids"].to('cuda')
                option_ids=batch["label_word_list"].to('cuda')
                
                attention_mask=batch["attention_mask"].to('cuda')
                token_type_ids=batch["token_type_ids"].to('cuda')
                labels=batch["labels"].to('cuda')
                
                
                
                #computing gradients for the slow weights!
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits  = outputs.logits.contiguous()

                
                
                indices = torch.where(token_type_ids[..., 1:] == 1)
                
               
                logits = logits[indices]
                
               
                nlogits = []
                for i in range(len(input_ids)):
                    nlogits += [ logits[i, option_ids[i]] ]
                
                
                logits = torch.stack(nlogits, 0)
                
          
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                losses = torch.mean(loss_fct(logits, labels.view(-1)))
                losses.backward()

                counter += 1

                if counter == self.args.gradient_accumulation_steps:

                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    counter = 0

                    self.global_step += 1

                    # ----------------------------------------------------------------------
                    # BEGIN CHANGES.
                    # ----------------------------------------------------------------------

                    #print (self.args.evaluate_during_training, self.global_step, self.args.eval_steps)
                    if  self.global_step % self.args.eval_steps == 0:
                        output = self.evaluate(eval_dataset=self.eval_dataset).compute()
                        try:
                            objective = output['accuracy']
                        except:
                            objective = output['f1']
                            
                        if objective > self.objective:
                            print("Best dev result: {}".format(objective))
                            self.objective = objective
                            self.save_model(self.args.output_dir)
                        self.model.train()
                    # ----------------------------------------------------------------------
                    # END CHANGES.
                    # ---------------------------------------------------------------------- 


                    
                
        
    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        
        
        
        per_example_batch_size = eval_dataset.num_labels    
        eval_dataloader = self.get_eval_dataloader(eval_dataset=eval_dataset)

        self.model.eval()
        true_labels = []
        print ("----------------------- length of test set -----------", len(true_labels) )
        correct = 0.
        
        all_predictions = []
        for batch in eval_dataloader:
        
            input_ids=batch["input_ids"].to('cuda')
            option_ids=batch["label_word_list"].to('cuda')
            
            attention_mask=batch["attention_mask"].to('cuda')
            token_type_ids=batch["token_type_ids"].to('cuda')
            labels=batch["labels"].to('cuda')
            
            
                        
            #evaluate now
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.contiguous()

            indices = torch.where(token_type_ids[..., 1:] == 1)
            
            logits = logits[indices]
            nlogits = []
            for i in range(len(input_ids)):
                nlogits += [ logits[i, option_ids[i]] ]
            
            logits = torch.stack(nlogits, 0)
            
            predictions = torch.argmax(logits, axis=-1)
            
            all_predictions += predictions.detach().cpu().numpy().tolist()
            true_labels += labels.detach().cpu().numpy().tolist()
            
            
        if eval_dataset.task_name.lower() not in [ 'qqp', 'mrpc' ]: 
            metric = load_metric("accuracy")
        else:
            metric = load_metric("f1")
            
        
        
        metric.add_batch(predictions=all_predictions, references=true_labels)       
        return metric