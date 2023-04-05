bash

export BERT_BASE_DIR=./models/roberta_wwm_ext

export DATASET=./datasets/weibo_comments

python run_classifier.py \
  --data_dir=$DATASET \
  --task_name=weibo_comments \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --output_dir=./output/ \
  --export_model_dir=./output/weibo_comments/export_models \
  --do_train=true \
  --do_eval=true \
  --do_predict=False  \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=2.0


#执行步骤：
'''
1、 export BERT_BASE_DIR=./models/roberta_wwm_ext

    export DATASET=./datasets/sensitive_identify

    python run_classifier.py \
        --data_dir=$DATASET \
        --task_name=sensitive_identify \
        --vocab_file=$BERT_BASE_DIR/vocab.txt \
        --bert_config_file=$BERT_BASE_DIR/bert_config.json \
        --output_dir=./output/ \
        --export_model_dir=./output/sensitive_identify/export_models \
        --do_train=true \
        --do_eval=true \
        --do_predict=False  \
        --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
        --max_seq_length=128 \
        --train_batch_size=32 \
        --learning_rate=2e-5 \
        --num_train_epochs=2.0

2、 导出模型
    python run_classifier.py \
        --data_dir=$DATASET \
        --task_name=sensitive_identify \
        --vocab_file=$BERT_BASE_DIR/vocab.txt \
        --bert_config_file=$BERT_BASE_DIR/bert_config.json \
        --output_dir=./output/ \
        --export_model_dir=./output/sensitive_identify/export_models \
        --do_train=false \
        --do_eval=false \
        --do_predict=true  \
        --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
        --max_seq_length=128 \
        --train_batch_size=32 \
        --learning_rate=2e-5 \
        --num_train_epochs=2.0
'''