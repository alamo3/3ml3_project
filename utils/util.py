import os

import torch
import torchvision
from tqdm import tqdm

from kitti_loader import dataset_kitti
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import torch.nn as nn
import sklearn.metrics as m
import json
import matplotlib.pyplot as plt

from params import *

from utils.labels import *


def save_checkpoint(state, filename="checkpoint_model.pth.tar"):
    print("Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    model.load_state_dict(checkpoint["state_dict"])


def get_training_loader(kitti_dataset_dir, batch_size, train_transform, val_transform, num_workers, pin_memory=True):
    dataset = dataset_kitti.KittiDataset(kitti_dataset_dir, train_transform)
    dataset_val = dataset_kitti.KittiDataset(kitti_dataset_dir, val_transform)

    ds_size = len(dataset)
    indices = list(range(ds_size))
    np.random.shuffle(indices)

    val_split_index = int(np.floor(0.2 * ds_size))
    train_idx, val_idx = indices[val_split_index:], indices[:val_split_index]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(dataset=dataset, shuffle=False, batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(dataset=dataset_val, shuffle=False, batch_size=batch_size, sampler=val_sampler,
                            num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader


def get_output_classes(pred):
    soft_max = nn.Softmax2d()
    out_classes_prob = soft_max(pred)

    out_classes = torch.argmax(out_classes_prob, dim=1)

    return out_classes


def validation_metrics(model, loader, num_classes, loss_fn, calculate_metrics=False, use_important_labels_only=True):
    loop = tqdm(loader)
    model.eval()

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    num_images_tested = 0

    y_true = []
    y_pred = []

    imp_labels = important_classes

    loss = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.float().to(device=DEVICE)
        targets = targets.squeeze(1).long().to(device=DEVICE)
        predictions = model(data)
        loss += loss_fn(predictions, targets).item()

        pred_classes = get_output_classes(predictions)

        targets_flat = torch.flatten(targets).tolist()
        pred_flat = torch.flatten(pred_classes).tolist()

        y_true.extend(targets_flat)
        y_pred.extend(pred_flat)

        for i in range(len(targets_flat)):
            confusion_matrix[targets_flat[i], pred_flat[i]] += 1

        num_images_tested += BATCH_SIZE

    if calculate_metrics:
        target_names = [get_class_name_for_id(i) for i in imp_labels]

        class_report = m.classification_report(y_true=y_true, y_pred=y_pred, output_dict=True, labels=imp_labels,
                                               target_names=target_names, zero_division=1)

        print(json.dumps(class_report, indent=4))

        accuracy_score = class_report['weighted avg']['precision']
        f1_score_micro = class_report['micro avg']['f1-score']
        f1_score_macro = class_report['macro avg']['f1-score']

        print('Balanced accuracy score', accuracy_score)
        print('Dice score Micro average', f1_score_micro)
        print('Dice score Macro Average', f1_score_macro)
        print('Loss function', loss / len(loader))

        return class_report, accuracy_score, f1_score_micro, f1_score_macro, loss / len(loader)
    else:
        return loss / len(loader)


def create_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def create_and_save_epoch_plots(loss_val, loss_train):
    fig, axs = plt.subplots(2)
    axs[0].plot(loss_train)
    axs[0].set(xlabel="Epoch", ylabel="Training Loss")


    axs[1].plot(loss_val)
    axs[1].set(xlabel="Epoch", ylabel="Validation Loss")
    plt.savefig(output_folder+'train_val_loss.png')
