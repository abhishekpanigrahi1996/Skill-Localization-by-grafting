import torch
import transformers
import torch.nn as nn
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import math 
from transformers import get_constant_schedule_with_warmup, get_constant_schedule, get_linear_schedule_with_warmup
from datasets import load_metric    

class graft_Trainer(nn.Module):
    def __init__(self, model_trainer):
        
        super(graft_Trainer, self).__init__()
        self.trainer = model_trainer
        self.model   = self.trainer.model
        self.args    = self.trainer.args
        
        self.trainer.select_trainable_parameters()
        self.params  = self.trainer.params
        
        
    
    ########################################################################################################################
    #We need to store pre-trained and fine-tuned model weights as well (Inefficient, need to think more on this)
    ########################################################################################################################
    def augment_models (self, pretrained_model, finetuned_model, model_args, device):   
        self.pretrained_model = pretrained_model
        self.finetuned_model  = finetuned_model
        self.device = device
        self.model_args = model_args
    
    ########################################################################################################################
    #The following function initializes the mask for grafting optimization
    ########################################################################################################################
    def create_binary_masks(self):
        
        self.trainable_name = []
        self.trainable_parameters = []
       
        for n in self.params: 
            self.trainable_name += [n]
            p = self.params[n]
            self.trainable_parameters += [ torch.rand_like( p.data, device=self.device, requires_grad=False) ] 
        
        
        self.num_params = sum([p.numel() for p in self.trainable_parameters])  

        self.grad_directions = []
        for counter in range(len(self.trainable_name)):
            for pre_n, pre_p in self.pretrained_model.named_parameters():
                if pre_n == self.trainable_name[counter]: pretensor = pre_p


            for fine_n, fine_p in self.finetuned_model.named_parameters():
                if fine_n == self.trainable_name[counter]: finetensor = fine_p
                    
                    
            self.grad_directions += [ (finetensor - pretensor).detach() ]        
    ########################################################################################################################

    
    ########################################################################################################################
    #The following function resets the model to pretrained model weights
    ########################################################################################################################   
    def reset_model(self):
        sigmoid = torch.nn.Sigmoid()
        for counter in range(len(self.trainable_name)):
            for pre_n, pre_p in self.pretrained_model.named_parameters():
                if pre_n == self.trainable_name[counter]: pretensor = pre_p.to(self.device)



            with torch.no_grad():   
                for n, p in self.model.named_parameters():    
                    if n == self.trainable_name[counter]: 
                    #    frac = sigmoid(trainable_parameters[counter] - sigmoid_bias)
                        p += ( pretensor - p )
    ########################################################################################################################
    
    
    

    
    ########################################################################################################################
    #The following function gets the grafted model with a given mask (or the current trainable parameters)
    ########################################################################################################################
    def interpolate_model(self, round_=False, mask=None):  
        sigmoid = torch.nn.Sigmoid()
        sigmoid_bias = self.args.sigmoid_bias
        for counter in range(len(self.trainable_name)):
            for pre_n, pre_p in self.pretrained_model.named_parameters():
                if pre_n == self.trainable_name[counter]: pretensor = pre_p.to(self.device)


            for fine_n, fine_p in self.finetuned_model.named_parameters():
                if fine_n == self.trainable_name[counter]: finetensor = fine_p.to(self.device)

            with torch.no_grad():            
                for n, p in self.model.named_parameters():    
                    if n == self.trainable_name[counter]: 
                        if mask is not None:
                            frac = self.basepatch[counter] + (1. - 2. * self.basepatch[counter]) * mask[counter]
                        else:    
                            frac = self.basepatch[counter] + (1. - 2. * self.basepatch[counter]) * sigmoid(self.trainable_parameters[counter] - sigmoid_bias) 
                            if round_:
                                frac = torch.round(frac)
                        p += frac * ( finetensor - pretensor ) 
    ########################################################################################################################                   
    
    ########################################################################################################################
    #This function creates the basepatch used for initializing the mask for optimization!
    #If mask_path == "highest_movement", we simply pick the parameters that have moved the most during training
    ########################################################################################################################
    def create_basepatch(self):
        sigmoid = torch.nn.Sigmoid()
        sigmoid_bias = self.args.sigmoid_bias
        num_params = self.num_params
        mask_path = self.model_args.mask_path 
        sparsity_level =  self.model_args.sparsity_level
        
        
        #If mask is already stored somewhere, I simply load it!
        if mask_path != "highest_movement":
            basepatch = torch.load(mask_path, map_location=self.device)

            
            total = max([ torch.amax(p) for p in basepatch ])
            #if the max value is greater than 1., it means we have received masks without sigmoid
            if total > 1.:
                basepatch[mask_counter] = [ sigmoid( p - sigmoid_bias ) for p in basepatch ]
            
            basepatch = [ torch.round( torch.clip (p, 0., 1.) )  for p in basepatch ]
            print ('Total parameters in my graft: ', sum([ torch.sum(p*p) / (1. * num_params) for p in basepatch ]))
            
            
        elif mask_path == "highest_movement":

            threshold = int(sparsity_level * num_params)
            
            best_top = np.zeros(threshold)

            consider = self.grad_directions

            for p in consider:
                arr = np.absolute(np.ndarray.flatten(p.detach().cpu().numpy()))
                all_magnitude = np.concatenate( [np.absolute(arr), best_top] )
                best_top = -np.sort(-all_magnitude)[:threshold]  


            all_magnitude = np.asarray(best_top)  
            

            threshold = np.sort(all_magnitude)[ 0 ]

            basepatch = [torch.zeros_like(p, requires_grad=False) for p in self.trainable_parameters]


            for p, q in zip(consider, basepatch):
                q[torch.absolute(p) > threshold] = 1.

            print ('Total parameters in my stitch: ', sum([ torch.sum(p*p) / (1. * num_params) for p in basepatch ]))
        else:
            raise NotImplementedError("Not Implemented!")
            
        self.basepatch = basepatch
    ########################################################################################################################
    
    
    
    ######################################################################################################################## 
    #For debugging, I re-defined evaluation here!
    ########################################################################################################################   
    def evaluate(self, dataloader, task_name, mode='dev'):
        if task_name.lower() not in [ 'qqp', 'mrpc' ]: 
            metric = load_metric("accuracy")
        else:
            metric = load_metric("f1")
            
        self.model.eval()
        hidden_states = []
        counter = 0 
        device = self.device
        for batch in dataloader:
            with torch.no_grad():
                if 'prompt' in self.model_args.few_shot_type :
                    loss, outputs = self.model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), mask_pos=batch["mask_pos"].to(device), labels=batch["labels"].to(device))
                elif ('finetune' in self.model_args.few_shot_type and  self.model_args.use_CLS_linearhead == 1) : 
                    loss, outputs = self.model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), labels=batch["labels"].to(device))
                elif 'finetune' in self.model_args.few_shot_type :
                    outputs = self.model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device)).logits
                    
            predictions = torch.argmax(outputs, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
            counter += 1
            if mode=='train' and counter >= self.args.gradient_accumulation_steps: break
            
        return metric
    ########################################################################################################################

    
    ########################################################################################################################
    #Main function that trains our graft!
    #We donot use an optimizer to train the graft, but compute the gradient w.r.t. the mask ourselves
    ########################################################################################################################    
    def train_graft (self, \
                     train_dataloader, \
                     valid_dataloader, \
                     eval_dataset, \
                     autoregressive, \
                     task_name, \
                    ):
        
        baseline = 0.  
        loss_fct = torch.nn.CrossEntropyLoss()
        first_batch = 0
        sigmoid = torch.nn.Sigmoid()
        checkpoint_location = self.model_args.checkpoint_location
        
        
        device = self.device
        lr = self.args.learning_rate
        sigmoid_bias = self.args.sigmoid_bias
        num_params = self.num_params
        
        for _ in tqdm( range(int(self.args.num_train_epochs)), 'Training the mask' ):
            total_grad = []
            
            first_batch = 0
            self.interpolate_model()

            for batch in train_dataloader:
                if 'prompt' in self.model_args.few_shot_type :
                    loss, outputs = self.model(input_ids=batch['input_ids'].to(device), \
                                               attention_mask=batch['attention_mask'].to(device), \
                                               mask_pos=batch["mask_pos"].to(device), \
                                               labels=batch["labels"].to(device), \
                                              )
                    
                elif ('finetune' in self.model_args.few_shot_type and  self.model_args.use_CLS_linearhead == 1) : 
                    loss, outputs = self.model(input_ids=batch['input_ids'].to(device), \
                                               attention_mask=batch['attention_mask'].to(device), \
                                               labels=batch["labels"].to(device), \
                                              )   
                    
                elif 'finetune' in self.model_args.few_shot_type :
                    loss = self.model(input_ids=batch['input_ids'].to(device), \
                                      attention_mask=batch['attention_mask'].to(device), \
                                      labels=batch['labels'].to(device), \
                                     ).loss
                    
                elif 'autoregressive' in self.model_args.few_shot_type :
                    input_ids=batch["input_ids"].to(device)
                    option_ids=batch["label_word_list"].to(device)

                    attention_mask=batch["attention_mask"].to(device)
                    token_type_ids=batch["token_type_ids"].to(device)
                    labels=batch["labels"].to(device)



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
                    loss = torch.mean(loss_fct(logits, labels.view(-1)))



                loss.backward()
                
                for n, p in self.model.named_parameters() :
                    if n in self.trainable_name :
                        if p.grad is None: print (n)

                grad = [p.grad.detach().clone() for n, p in self.model.named_parameters() if n in self.trainable_name]
                self.model.zero_grad()
                grad = [ g * p.to(device) for (g, p) in zip(grad, self.grad_directions) ]


                if first_batch == 0:
                    total_grad = [lr * g for g in grad]
                else:
                    total_grad = [ p + lr * g for (p, g) in zip( total_grad, grad ) ]
                first_batch += 1 
                #restrict the number of loops
                if first_batch >= self.args.gradient_accumulation_steps: 
                    break

            total_grad = [ p / (1. * first_batch) for p in total_grad ]    
            self.reset_model()
       
            #Take the gradient step
            with torch.no_grad():
                for p, (g, s) in zip(self.trainable_parameters, zip(total_grad, self.basepatch)):
                    p -=  ( (1. - 2.*s) * g * sigmoid(p - sigmoid_bias) * (1. - sigmoid(p - sigmoid_bias)) )
         
            ######### Evaluation of current mask ###########
            self.interpolate_model(round_=True)
            if task_name.lower() not in [ 'qqp', 'mrpc' ]: key = "accuracy"
            else: key = "f1"
                
            if autoregressive:
                tr  = self.trainer.evaluate(train_dataset).compute()[key] 
                val = self.trainer.evaluate(eval_dataset).compute()[key] 
            else:            
                tr  = self.evaluate(train_dataloader, task_name, mode='train').compute()[key]
                val = self.evaluate(valid_dataloader, task_name).compute()[key]

            #store the mask with the best train + validation score
            bs_compare = val + tr

            if bs_compare > baseline:
                torch.save(self.trainable_parameters, checkpoint_location)
                baseline = bs_compare
               
            self.reset_model()      

    ########################################################################################################################
    
        
        
        
        
        