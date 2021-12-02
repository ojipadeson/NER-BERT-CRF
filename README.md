# NER implementation with BERT and CRF model

### Modified from:
> Zhibin Lu

This is a named entity recognizer based on [BERT Model(pytorch-pretrained-BERT)](https://github.com/huggingface/pytorch-pretrained-BERT) and CRF.

Someone construct model with BERT, LSTM and CRF, like this [BERT-BiLSTM-CRF-NER](https://github.com/FuYanzhe2/Name-Entity-Recognition/tree/master/BERT-BiLSTM-CRF-NER), but in theory, the BERT mechanism has replaced the role of LSTM, so I think LSTM is redundant.

For the performance, BERT+CRF is always a little better than single BERT in my experience.

## Requirements
- python 3.6
- pytorch 1.0.0
- [pytorch-pretrained-bert 0.4.0](https://github.com/huggingface/transformers/releases/tag/v0.4.0)
## Overview
The NER_BERT_CRF.py include 2 model:
- model 1:
  - This is just a pretrained BertForTokenClassification, For a comparision with my BERT-CRF model
- model 2:
  - A pretrained BERT with CRF model.
- data set
  - [CoNLL-2003](https://github.com/Franck-Dernoncourt/NeuroNER/tree/master/neuroner/data/conll2003/en)
### Parameters
- NER_labels = ['X', '[CLS]', '[SEP]', 'O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
- max_seq_length = 180
- batch_size = 32
- learning_rate = 5e-5
- weight_decay = 1e-5
- learning_rate for CRF and FC: 8e-5 
- weight_decay for CRF and FC: 5e-6
- total_train_epochs = 20
- bert_model_scale = 'bert-base-cased'
- do_lower_case = False

### References
- [Bert paper](https://arxiv.org/abs/1810.04805)
- [Bert with PyTorch implementation](https://github.com/huggingface/pytorch-pretrained-BERT)
- [ericput/Bert-ner](https://github.com/ericput/bert-ner)
- [CoNLL-2003 data set](https://github.com/Franck-Dernoncourt/NeuroNER/tree/master/neuroner/data/conll2003/en)
- [Kyubyong/bert_ner](https://github.com/Kyubyong/bert_ner)
