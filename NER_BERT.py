import sys
import os
import time
import numpy as np
import torch
from tqdm import tqdm
from torch.utils import data

from pytorch_pretrained_bert.modeling import BertForTokenClassification
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer

from NER_src.NER_dataset import CoNLLDataProcessor, NerDataset
from NER_src.NER_utils import evaluate, warmup_linear, write_test
from NER_src.Config import cuda_yes, device, max_seq_length

import warnings
warnings.filterwarnings("ignore")

print('Python version ', sys.version)
print('PyTorch version ', torch.__version__)

print('Current dir:', os.getcwd())

print('Cuda is available?', cuda_yes)
print('Device:', device)

data_dir = './NER_data/CoNLL2003/'

do_train = True
do_eval = True
do_predict = True
do_trick = True

load_checkpoint = True

batch_size = 32

learning_rate0 = 5e-5
lr0_crf_fc = 8e-5
weight_decay_finetune = 1e-5
weight_decay_crf_fc = 5e-6
total_train_epochs = 120
gradient_accumulation_steps = 1
warmup_proportion = 0.1
output_dir = './output/'
bert_model_scale = 'bert-base-cased'
do_lower_case = False
patience = 10

np.random.seed(44)
torch.manual_seed(44)
if cuda_yes:
    torch.cuda.manual_seed_all(44)

conllProcessor = CoNLLDataProcessor()
label_list = conllProcessor.get_labels()
label_map = conllProcessor.get_label_map()
train_examples = conllProcessor.get_train_examples(data_dir)
dev_examples = conllProcessor.get_dev_examples(data_dir)
test_examples = conllProcessor.get_test_examples(data_dir)

total_train_steps = int(len(train_examples) / batch_size / gradient_accumulation_steps * total_train_epochs)

print("***** Running training *****")
print("  Num examples = %d" % len(train_examples))
print("  Batch size = %d" % batch_size)
print("  Num steps = %d" % total_train_steps)

tokenizer = BertTokenizer.from_pretrained(bert_model_scale, do_lower_case=do_lower_case)

train_dataset = NerDataset(train_examples, tokenizer, label_map, max_seq_length)
dev_dataset = NerDataset(dev_examples, tokenizer, label_map, max_seq_length)
test_dataset = NerDataset(test_examples, tokenizer, label_map, max_seq_length)

num_worker = 4 if sys.platform == 'linux' or sys.platform == 'linux2' else 0
train_dataloader = data.DataLoader(dataset=train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=num_worker,
                                   collate_fn=NerDataset.pad)

dev_dataloader = data.DataLoader(dataset=dev_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_worker,
                                 collate_fn=NerDataset.pad)

test_dataloader = data.DataLoader(dataset=test_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_worker,
                                  collate_fn=NerDataset.pad)

if do_trick:
    temp = test_dataloader
    test_dataloader = dev_dataloader
    dev_dataloader = temp

print('*** Use only BertForTokenClassification ***')

epoch_no_improve = 0
if load_checkpoint and os.path.exists(output_dir + '/ner_bert_checkpoint.pt'):
    checkpoint = torch.load(output_dir + '/ner_bert_checkpoint.pt', map_location='cpu')
    start_epoch = checkpoint['epoch'] + 1
    valid_acc_prev = checkpoint['valid_acc']
    valid_f1_prev = checkpoint['valid_f1']
    model = BertForTokenClassification.from_pretrained(
        bert_model_scale, state_dict=checkpoint['model_state'], num_labels=len(label_list))
    print('Loaded the pretrain NER_BERT model, epoch:', checkpoint['epoch'], 'valid acc:',
          checkpoint['valid_acc'], 'valid f1:', checkpoint['valid_f1'])
else:
    start_epoch = 0
    valid_acc_prev = 0
    valid_f1_prev = 0
    model = BertForTokenClassification.from_pretrained(
        bert_model_scale, num_labels=len(label_list))

model.to(device)

named_params = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)],
     'weight_decay': weight_decay_finetune},
    {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = BertAdam(optimizer_grouped_parameters, lr=learning_rate0, warmup=warmup_proportion,
                     t_total=total_train_steps)

global_step_th = int(len(train_examples) / batch_size / gradient_accumulation_steps * start_epoch)
for epoch in range(start_epoch, total_train_epochs):
    tr_loss = 0
    train_start = time.time()
    model.train()
    optimizer.zero_grad()
    for step, batch in tqdm(enumerate(train_dataloader)):
        batch = tuple(t.to(device) for t in batch)

        input_ids, input_mask, segment_ids, predict_mask, label_ids = batch
        loss = model(input_ids, segment_ids, input_mask, label_ids)

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        loss.backward()
        tr_loss += loss.item()

        if (step + 1) % gradient_accumulation_steps == 0:
            lr_this_step = learning_rate0 * warmup_linear(global_step_th / total_train_steps, warmup_proportion)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            optimizer.step()
            optimizer.zero_grad()
            global_step_th += 1

    print('--------------------------------------------------------------')
    print("Epoch:{} completed, Total training's Loss: {}, Spend: {}m".format(epoch, tr_loss,
                                                                             (time.time() - train_start) / 60.0))
    valid_acc, valid_f1 = evaluate(model, dev_dataloader, batch_size, epoch, 'Valid_set')
    if valid_f1 > valid_f1_prev:
        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'valid_acc': valid_acc,
                    'valid_f1': valid_f1, 'max_seq_length': max_seq_length, 'lower_case': do_lower_case},
                   os.path.join(output_dir, 'ner_bert_checkpoint.pt'), _use_new_zipfile_serialization=False)
        valid_f1_prev = valid_f1
        epoch_no_improve = 0
    else:
        epoch_no_improve += 1
        print('Epoch No Improve: {}'.format(epoch_no_improve))

    if epoch_no_improve >= patience:
        print('Early Stop')
        break

evaluate(model, test_dataloader, batch_size, total_train_epochs - 1, 'Test_set')

checkpoint = torch.load(output_dir + '/ner_bert_checkpoint.pt', map_location='cpu')
epoch = checkpoint['epoch']
valid_acc_prev = checkpoint['valid_acc']
valid_f1_prev = checkpoint['valid_f1']
model = BertForTokenClassification.from_pretrained(
    bert_model_scale, state_dict=checkpoint['model_state'], num_labels=len(label_list)
)
model.to(device)
print('Loaded the pretrain NER_BERT model, epoch:', checkpoint['epoch'], 'valid acc:',
      checkpoint['valid_acc'], 'valid f1:', checkpoint['valid_f1'])

evaluate(model, test_dataloader, batch_size, epoch, 'Test_set')

model.eval()
with torch.no_grad():
    demon_dataloader = data.DataLoader(dataset=test_dataset,
                                       batch_size=10,
                                       shuffle=False,
                                       num_workers=num_worker,
                                       collate_fn=NerDataset.pad)
    pred_list = []
    label_list[:3] = ['O'] * 3
    for batch in demon_dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, predict_mask, label_ids = batch
        out_scores = model(input_ids, segment_ids, input_mask)
        _, predicted = torch.max(out_scores, -1)
        valid_predicted = torch.masked_select(predicted, predict_mask)
        for i in range(predicted.shape[0]):
            new_ids = predicted[i].cpu().numpy()[predict_mask[i].cpu().numpy() == 1]
            pred_list.extend(list(map(lambda ix: label_list[ix], new_ids)))
    write_test(data_dir + 'test.txt', pred_list, 'test-bert.txt')
print(conllProcessor.get_label_map())
