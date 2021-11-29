CUDA_VISIBLE_DEVICES=0 python train.py\
    --language english\
    --model_name bert-base-multilingual-cased\
    --train_data preprocessing/bn_tokenized.txt\
    --number_of_gpu 1\
    --max_len 256\
    --batch_size_per_gpu 16\
    --gradient_accumulation_steps 2\
    --effective_batch_size 256\
    --learning_rate 1e-4\
    --total_steps 150010\
    --print_every 500\
    --save_every 10000\
    --ckpt_save_path ./ckpt/tacl_english/
