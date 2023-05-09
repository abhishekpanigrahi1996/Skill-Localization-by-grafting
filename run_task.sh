seed=0; 
bs=2; 
lr=1e-3; 
model="roberta-base";
TASK='SST-2'; 
max_step=1000;

for K in 16; do
    TAG=exp \
    TYPE=prompt \
    TASK=$TASK \
    K=$K \
    BS=$bs \
    LR=$lr \
    SEED=$seed \
    modelseed=0 \
    uselmhead=1 \
    useCLS=0 \
    max_step=$max_step \
    fixhead=True \
    fixembeddings=True \
    MODEL=$model \
    train_bias_only=False \
    MODELNAME=$model\
    bash run_experiment.sh;
done