# export BERT_BASE_DIR=/root/projects/transformers/src/transformers/models/bert/chinese_L-12_H-768_A-12
export BERT_BASE_DIR=/root/projects/transformers/src/transformers/models/a_simple_bert/bert-base-uncased

python convert_bert_original_tf_checkpoint_to_pytorch.py \
 --tf_checkpoint_path $BERT_BASE_DIR/bert_model.ckpt \
 --bert_config_file $BERT_BASE_DIR/bert_config.json \
 --pytorch_dump_path ./torch_pretrain/bert_uncased.bin