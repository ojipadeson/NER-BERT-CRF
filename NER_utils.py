import numpy as np
import time

import torch
from sklearn.metrics import classification_report

from Config import device


def f1_score(y_true, y_pred):
    ignore_id = 3

    num_proposed = len(y_pred[y_pred > ignore_id])
    num_correct = (np.logical_and(y_true == y_pred, y_true > ignore_id)).sum()
    num_gold = len(y_true[y_true > ignore_id])

    try:
        precision = num_correct / num_proposed
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = num_correct / num_gold
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        if precision * recall == 0:
            f1 = 1.0
        else:
            f1 = 0

    return precision, recall, f1


def ner_evaluation(true_label, predicts):
    ignore_id = 2
    report_dict = classification_report(true_label[true_label > ignore_id], predicts[predicts > ignore_id],
                                        digits=4, output_dict=True)
    report = classification_report(true_label[true_label > ignore_id], predicts[predicts > ignore_id],
                                   digits=4)
    print(report)
    return report_dict['macro avg']['f1-score']


def evaluate(model, predict_dataloader, batch_size, epoch_th, dataset_name, use_crf=False):
    model.eval()
    all_preds = []
    all_labels = []
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in predict_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, predict_mask, label_ids = batch

            if not use_crf:
                out_scores = model(input_ids, segment_ids, input_mask)
                _, predicted = torch.max(out_scores, -1)
                valid_predicted = torch.masked_select(predicted, predict_mask)
                valid_label_ids = torch.masked_select(label_ids, predict_mask)
            else:
                _, predicted_label_seq_ids = model(input_ids, segment_ids, input_mask)
                valid_predicted = torch.masked_select(predicted_label_seq_ids, predict_mask)
                valid_label_ids = torch.masked_select(label_ids, predict_mask)

            all_preds.extend(valid_predicted.tolist())
            all_labels.extend(valid_label_ids.tolist())
            total += len(valid_label_ids)
            correct += valid_predicted.eq(valid_label_ids).sum().item()

    test_acc = correct / total
    print(np.array(all_labels).shape, '\n', np.array(all_preds).shape)
    print(np.array(all_labels).min, '\n', np.array(all_labels).max)
    print(np.array(all_preds).min, '\n', np.array(all_preds).max)
    f1 = ner_evaluation(np.array(all_labels), np.array(all_preds))
    return test_acc, f1


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x
