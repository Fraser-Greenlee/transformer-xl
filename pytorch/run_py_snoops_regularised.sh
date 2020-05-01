

OMP_NUM_THREADS=1 python train.py \
    --cuda \
    --data ../data/py_snoops/ \
    --dataset py_snoops \
    --adaptive \
    --gpu0_bsz 1 \
    --fp16 \
    --dynamic-loss-scale \
    --log-interval 50 \
    --eval-interval 400 \
    --d_model 400 \
    --n_head 10 \
    --d_head 40 \
    --warmup_step 3000 \
    --tgt_len 150 \
    --mem_len 150 \
    --eval_tgt_len 150 \
    --batch_size 32 \
    --seed 0 \
    --lr 0.00035 \
    --max_step 125000 \
    --dropouti 0.6 \
    --dropouto 0.5 \
    --dropoute 0.2 \
    --dropout 0.2 \
    --dropatt 0.2 \
    --d_inner 900 \
    --n_layer 16
