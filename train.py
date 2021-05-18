from modeling import Bert
import torch
import torch.nn as nn
from collections import OrderedDict
from bert_pretrain import BertPretrain

def get_dataloader(path, tokenizer):
    
    from datasets import load_dataset
    dataset = load_dataset('glue', 'mrpc')

    def encode(examples):
        return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length', max_length=128)
    dataset = dataset.map(encode, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
    train_loader = torch.utils.data.DataLoader(dataset['train'], batch_size=32)
    validation_loader = torch.utils.data.DataLoader(dataset['validation'], batch_size=32)
    test_loader = torch.utils.data.DataLoader(dataset['test'], batch_size=32)

    return train_loader, validation_loader, test_loader

def eval(model,test_loader,device,loss_fn):
    total_loss=0
    total_num=0
    correct=0
    with torch.no_grad():
        for batch in test_loader:
            input_ids=batch['input_ids'].to(device)
            segment_ids=batch['token_type_ids'].to(device)
            cls_probs = model(input_ids=input_ids, position_ids=None, segment_ids=segment_ids, attention_mask=None)
            
            pred_labels=cls_probs.argmax(-1)
            labels=batch['label'].to(device)
            correct += (pred_labels == labels).float().sum()
            
            loss = loss_fn(cls_probs, labels)
            total_loss+=input_ids.size()[0]*loss.item()
            
            total_num+=input_ids.size()[0]
            
    print('eval loss \t',total_loss/total_num)
    print('eval accuracy \t',correct/total_num)

def main():
    config_path = './model/bert-uncased.json'
    pretrain_model='bert'
    
    sequence_classification_tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'google/bert_uncased_L-12_H-768_A-12')
    train_loader, validation_loader, test_loader = get_dataloader(config_path, sequence_classification_tokenizer)

    model = BertClassifier(config_path)
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 2e-5
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model=model.to(device)
    from torch.utils.tensorboard import SummaryWriter
    import numpy as np
    print(model)
    writer = SummaryWriter(log_dir='/root/projects/summary')

    global_step=0
    for epoch  in range(3):
        for batch in train_loader:
            model.train()
            input_ids=batch['input_ids'].to(device)
            segment_ids=batch['token_type_ids'].to(device)
            cls_probs = model(input_ids=input_ids, position_ids=None, segment_ids=segment_ids, attention_mask=None)
            labels=batch['label'].to(device)
            loss = loss_fn(cls_probs, labels)
            loss.backward()
            optim.step()
            optim.zero_grad()
            writer.add_scalar('Loss/train', loss.item(), global_step)
            global_step+=1
            # if global_step %10==0:
            #     model.eval()
            #     eval(model, test_loader,device,loss_fn)
    model.eval()
    eval(model, test_loader,device,loss_fn)
    
if __name__ == '__main__':
    main()
