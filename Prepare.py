import os
import sys


def set_work_dir(local_path="ner_bert_crf", server_path="ner_bert_crf"):
    if sys.platform == 'linux' or sys.platform == 'linux2':
        if os.path.exists(os.getenv("HOME") + '/' + local_path):
            os.chdir(os.getenv("HOME") + '/' + local_path)
        elif os.path.exists(os.getenv("HOME") + '/' + server_path):
            os.chdir(os.getenv("HOME") + '/' + server_path)
        else:
            raise Exception('Set work path error!')
    else:
        os.chdir('C:\\FUDAN\\Grade4_1\\NLP\\Lab2\\ner_bert_crf')


def get_data_dir(local_path="ner_bert_crf", server_path="ner_bert_crf"):
    if sys.platform == 'linux' or sys.platform == 'linux2':
        if os.path.exists(os.getenv("HOME") + '/' + local_path):
            return os.getenv("HOME") + '/' + local_path
        elif os.path.exists(os.getenv("HOME") + '/' + server_path):
            return os.getenv("HOME") + '/' + server_path
        else:
            raise Exception('get data path error!')
    else:
        return 'C:\\FUDAN\\Grade4_1\\NLP\\Lab2\\ner_bert_crf'
