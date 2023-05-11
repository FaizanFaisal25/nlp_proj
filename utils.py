import os
import random
import logging

import torch
import numpy as np

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, multilabel_confusion_matrix
import matplotlib.pyplot as plt


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(labels, preds, output_matrix, label_list):
    assert len(preds) == len(labels)
    results = dict()

    print("CONFUSION MATRIX")
    cm = multilabel_confusion_matrix(labels, preds)
    fig, ax = plt.subplots(4, 7, figsize=(100, 50))
    for i in range(4):
        for j in range(7):
            disp = ConfusionMatrixDisplay(confusion_matrix=cm[i*7+j], display_labels=[label_list[i*7+j], "Not " + label_list[i*7+j]])
            disp.plot(ax=ax[i, j], values_format='d')
    plt.savefig(output_matrix+"_all"+".png")

    main_cm = np.zeros((2, 2))
    for i in range(len(cm)):
        main_cm += cm[i]
    disp = ConfusionMatrixDisplay(confusion_matrix=main_cm, display_labels=["yes", "no"])
    disp.plot()
    plt.savefig(output_matrix+"_main"+".png")

    results["accuracy"] = accuracy_score(labels, preds)
    results["macro_precision"], results["macro_recall"], results[
        "macro_f1"], _ = precision_recall_fscore_support(
        labels, preds, average="macro")
    results["micro_precision"], results["micro_recall"], results[
        "micro_f1"], _ = precision_recall_fscore_support(
        labels, preds, average="micro")
    results["weighted_precision"], results["weighted_recall"], results[
        "weighted_f1"], _ = precision_recall_fscore_support(
        labels, preds, average="weighted")

    return results
