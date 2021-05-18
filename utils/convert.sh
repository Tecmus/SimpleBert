#以bert-base-uncased为例 地址https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip

export BERT_BASE_DIR=../model/uncased_L-12_H-768_A-12

python convert_bert_original_tf_checkpoint_to_pytorch.py \
 --tf_checkpoint_path $BERT_BASE_DIR/bert_model.ckpt \
 --bert_config_file $BERT_BASE_DIR/bert_config.json \
 --pytorch_dump_path ../output_model/bert_uncased.pt
 