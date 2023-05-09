modelbase='roberta-base';
task='SST-2';
seed=0;
K=16;
sparsitylevel=1e-07 ;
lr=1e7 ;
no_train=True; 


model_path="../ckpt_paths/log_noembed_SGD_graft/SST-2-prompt-16-0-roberta-base-4384-2-1e-3"
mask_path="highest_movement";
modelbase='roberta-base';

for K in 16; do
    TAG=exp \
    TYPE=prompt \
    TASK=$task \
    K=$K \
    LR=$lr \
    SEED=$seed \
    MODEL=$model_path \
    uselmhead=1 \
    useCLS=0\
    num_train_epochs=10 \
    mask_path=$mask_path \
    sparsitylevel=$sparsitylevel \
    pretrained_model=$modelbase \
    fixhead=True \
    fixembeddings=True \
    truncate_head=True\
    train_bias_only=False \
    no_train=$no_train \
    checkpoint_location="/tmp/mask_path"\
    bash run_graft_experiment.sh;
done
