python tokenize_data.py\
    --model_name bert-base-multilingual-cased\
    --raw_data_path ./train.hi\
    --output_dir ./bert-uncased/\
    --output_name hi_tokenized.txt

python tokenize_data.py\
    --model_name bert-base-multilingual-cased\
    --raw_data_path ./train.en\
    --output_dir ./bert-uncased/\
    --output_name en_tokenized.txt
