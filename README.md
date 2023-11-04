## Task-Specific Skill localization in Fine-tuned Language Models

This is the code repository for the paper: [Task-Specific Skill Localization in Fine-tuned Language Models](https://arxiv.org/abs/2302.06600) (Appeared in ICML'23).

## Quick Links

- [Task-Specific Skill localization in Fine-tuned Language Models](#task-specific-skill-localization-in-fine-tuned-language-models)
- [Quick Links](#quick-links)
- [Abstract](#overview)
 - [Overview on Grafting](#overview-on-grafting)
- [Fine-tuning and Grafting](#fine-tuning-and-grafting)
  - [Necessary Files](#necessary-files)
  - [Installation](#installation)
  - [Data](#data)
  - [Necessary Folders](#necessary-folders)
  - [Fine-tuning](#fine-tuning)
  - [Grafting](#grafting)
- [Bugs or Questions](#bugs-or-questions)
- [Citation](#citation)

## Abstract

Pre-trained language models can be fine-tuned to solve diverse NLP tasks, including in few-shot settings. Thus fine-tuning allows the model to quickly pick up task-specific “skills,” but there has been limited study of where these newlylearnt skills reside inside the massive model. This paper introduces the term skill localization for this problem and proposes a solution. Given the downstream task and a model fine-tuned on that task, a simple optimization is used to identify a very small subset of parameters (~0.01% of model parameters) responsible for (> 95%) of the model’s performance, in the sense that grafting
the fine-tuned values for just this tiny subset onto the pre-trained model gives a performance almost
as well as the fine-tuned model. While reminiscent of recent works on parameter-efficient
fine-tuning, the novel aspects here are that: (i) No further re-training is needed on the subset (unlike, say, with lottery tickets). (ii) Notable improvements are seen over vanilla fine-tuning with respect to calibration of predictions in-distribution
(40-90% error reduction) as well as the quality of predictions out-of-distribution (OOD). In models trained on multiple tasks, a stronger notion of skill localization is observed, where the sparse regions corresponding to different tasks are almost disjoint, and their overlap (when it happens) is a proxy for task similarity.



## Fine-tuning and Grafting

### Necessary Files
We provide a yml file that can be used to create a conda environment, containing all the necessary packages.

### Installation
Install necessary conda environment using 
conda env create -n icl_as_ft --file task_skill.yml

### Data
Please follow the repository [LM-BFF-main](https://github.com/princeton-nlp/LM-BFF#prepare-the-data) to download all data. In the remaining codes, we assume that there is a "data" folder containing all the necessary datasets.


### Necessary folders
Please make two new folders: log_files and ckpt_paths before running the code below. The results of fine-tuning and grafting are stored in a log file inside log_files (please see run_experiment.sh and run_graft_experiment.sh for the filenames). The model checkpoints are stored in ckpt_paths.


### Fine-tuning
To run grafting, we need to have a pre-trained model and a fine-tuned model (please refer to the overview section for grafting formulation). This section provides code to train roberta and gpt models on different downstream tasks.

Please refer to run_experiment.sh for the arguments that we use to train a model on a task. For all the SGD trained single task models in the paper, the current command line in run_task.sh suffices.

```bash
TAG=exp \
TYPE=$TYPE \
TASK=$TASK \
K=$K \
BS=$BS \
LR=$LR \
SEED=$seed \
modelseed=$modelseed \
uselmhead=$uselmhead \
useCLS=$useCLS \
max_step=$max_step \
fixhead=$fixhead \
fixembeddings=$fixembeddings \
MODEL=$model \
train_bias_only=$train_bias_only \
MODELNAME=$model\
bash run_experiment.sh   
```


* `TYPE`: There are three types of training
  * `finetune`: Linear Head-tuning of bert based models
  * `prompt`: Prompt-based fine-tuning of bert based models.
  * `autoregressive`: Prompt-based fine-tuning of gpt models.
* `TASK`: Please refer to run_experiments.sh for the exact task names to use
* `K`: Number of training samples per class
* `BS`: Batch size to use for training
* `LR`: Learning rate to use for training 
* `SEED`: Data seed
* `modelseed`: Model seed for training
* `uselmhead`: Whether to use Language model head (true for prompt-based training)
* `useCLS`: Whether to use CLS (for finetune experiments, if False, we train linear head on mask tokens)
* `train_bias_only`: Whether to train biases only
* `fixembeddings`: Fix embeddings? (helps for SGD based training)
* `fixhead`: Fix head? We fix LM head in all our prompt based training experiments
* `model`: roberta-base/gpt-2



### Grafting
With a pre-trained and fine-tuned model, we perform grafting to find the minimal set of parameters from the fine-tuned model that can be grafted onto the pre-trained model and still retain most of the model's final performance. 

Please refer to run_graft_experiment.sh for the arguments that we use to train a graft on a task. For all the SGD trained single task models in the paper, the current command line in run_graft_task.sh suffices.


```bash
TAG=exp \
TYPE=$TYPE \
TASK=$TASK \
K=$K \
LR=$lr \
SEED=$seed \
MODEL=$model_path \
modelseed=$modelseed \
uselmhead=$uselmhead \
useCLS=$useCLS \
num_train_epochs=100 \
mask_path=$mask_path \
sparsitylevel=$sparsitylevel \
pretrained_model=$modelbase \
fixhead=$fixhead \
fixembeddings=$fixembeddings \
truncate_head=True\
no_train=$no_train \
checkpoint_location=$location\
bash run_graft_experiment.sh;
```

* `TYPE`: Same as trained model 
* `TASK`: Please refer to run_experiments.sh for the exact task names to use
* `K`: Number of training samples per class
* `LR`: Learning rate to use for graft training 
* `SEED`: Data seed
* `modelseed`: Seed for graft training
* `uselmhead`: Same as trained model hyperparameter
* `useCLS`: Same as trained model hyperparameter
* `train_bias_only`: Same as trained model hyperparameter
* `fixembeddings`: Same as trained model hyperparameter
* `fixhead`: Same as trained model hyperparameter
* `MODEL`: Finetuned model path
* `pretrained_model`: roberta-base/gpt-2
* `sparsitylevel`: Sparsity Level of basepatch
* `mask_path`: Path to store the mask
* `no_train`: Whether to train a mask (if True, we upload mask from checkpoint_location)


## Bugs or Questions
If you have any questions related to the code, feel free to email Abhishek (`{ap34}@cs.princeton.edu`). If you encounter a problem or bug when using the code, you can also open an issue.


## Citation

Please cite our work if you make use of our code or our pre-computed kernels in your work:

```bibtex
@article{panigrahi2023task,
  title={Task-Specific Skill Localization in Fine-tuned Language Models},
  author={Panigrahi, Abhishek and Saunshi, Nikunj and Zhao, Haoyu and Arora, Sanjeev},
  journal={arXiv preprint arXiv:2302.06600},
  year={2023}
}
```
