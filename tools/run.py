import os
import os.path as osp
import argparse
import time
    
from glob import glob
import torch
import torch.nn as nn
import gc
import numpy as np
import fetal_monitor
from fetal_monitor.models.resnet import (ResidualBlock,
                                         ResNet)
from fetal_monitor.dataset.dataloader import data_loader
from fetal_monitor.utils.utils import calc_auc
from torch.utils.data import (Dataset,
                              DataLoader, 
                              ConcatDataset,
                              random_split)
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

# nni 
import nni
from nni.utils import merge_parameter

# metric
from sklearn import metrics
from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay

# sklearn class_weight
from sklearn.utils.class_weight import compute_class_weight

def save_auc_roc_fig(fname, label, predict):
    RocCurveDisplay.from_predictions(
        label,
        predict,
        name = f"Resnet vs the rest",
        color = "darkblue",
    )
    plt.plot([0,1], [0,1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC curve")
    plt.savefig(fname, dpi=100)

# Set tensor precision for print().
torch.set_printoptions(precision=2, sci_mode=False)

'''
TRAIN = [10450555,10657486,11057045,
         11632775,11661895,11762836,
         11781242,11977062,12059111,
         12120541,12130784,12144707,
         12168164,12177066,12182543,
         12189365,12206951,12218436,
         12227047,12230344,12250517,
         12257489,12258685,12261155,
         12273756]

VAL = [12250517,12257489,12258685,12261155,12273756]

CROSSVAL = [12231373,12239771,12240224,12241130]
'''

TRAIN = [10450555,10657486,11057045,
         12120541,11661895,11762836,
         11781242,12059111,12241130,
         12130784,12168164,12177066,
         12182543,12189365,12218436,
         12227047,12230344,12250517,
         12257489,12258685,12261155,
         12240224,12239771]

VAL = [12273756,12231373]

CROSSVAL = [11977062, 12144707, 12206951, 11632775]


def main():
    # params
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=32, help='batch size')
    parser.add_argument('--epoch', type=int, default=100, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--source', type=str, default=None, help='training/validation/test set path')

    # nni params
    arg, _ = parser.parse_known_args()
    tuner_params = nni.get_next_parameter()
    print('-------------------tuner_params-------------------\n', tuner_params)
    arg = merge_parameter(arg, tuner_params)
    print('-------------------arg_params-------------------\n', arg)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper Parameters
    num_epochs = arg.epoch
    num_classes = 2
    learning_rate = 0.00005
    resize_scala = 224
    batch_size = arg.bs

    # Dataset/DataLoader Preparation
    TRAIN_BASKET, VAL_BASKET, CROSSVAL_BASKET = \
        list(), list(), list()
    
    # Date-Time To save
    timestr = time.strftime("%Y%m%d-%H%M%S")

    for patient_id_path in glob(f'/home/sangwon/sources/resnet/tools/data/data_subtraction/*'): # f'/home/sangwon/sources/resnet/tools/data/*Data*0313*/*'
                                                                                                # f'/home/sangwon/sources/resnet/tools/data/data_subtraction/*'
        patient_id = int(osp.basename(patient_id_path))
        if patient_id in TRAIN:
            _dataset = ImageFolder(root=patient_id_path,
                                   transform=transforms.Compose([transforms.Resize([resize_scala, resize_scala]),
                                                                transforms.ToTensor(),
                                                                ]))
            TRAIN_BASKET.append(_dataset)
            
        elif patient_id in VAL:
            _dataset = ImageFolder(root=patient_id_path,
                                   transform=transforms.Compose([transforms.Resize([resize_scala, resize_scala]),
                                                                transforms.ToTensor(),
                                                                ]))
            VAL_BASKET.append(_dataset)
            
        elif patient_id in CROSSVAL:
            _dataset = ImageFolder(root=patient_id_path,
                                   transform=transforms.Compose([transforms.Resize([resize_scala, resize_scala]),
                                                                transforms.ToTensor(),
                                                                ]))
            CROSSVAL_BASKET.append(_dataset)
        else:
            raise Exception(f'{patient_id} not included in any group!')

    if (len(TRAIN_BASKET) == len(TRAIN) and
        len(VAL_BASKET) == len(VAL) and
        len(CROSSVAL_BASKET) == len(CROSSVAL)):
        
        train_dataset = ConcatDataset(TRAIN_BASKET)
        valid_dataset = ConcatDataset(VAL_BASKET)
        c_val_dataset = ConcatDataset(CROSSVAL_BASKET)
    else:
        raise Exception(f'There should be patient ids omitted!')

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)

    # fold_batch_size = batch_size
    global_loader = DataLoader(train_dataset,
                               batch_size=int(len(train_dataset)),
                               shuffle=True,
                               num_workers=4)
    
    # Loss
    for gi, (images, labels) in enumerate(global_loader):
        # Move tensors to the configured device
        print(f'global loader index : {gi}')
        images = images.to(device)
        labels = labels.to(device)
        class_weights = compute_class_weight(class_weight='balanced',
                                             classes=np.unique(labels.cpu().detach().numpy()),
                                             y=labels.cpu().detach().numpy())
        class_weights = torch.Tensor(class_weights)
        criterion_weighted = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print(f'global weight : {class_weights}')

    # model
    model = ResNet(ResidualBlock, [3, 2, 6, 3], num_classes).to(device)
                   # in_channels, out_channels, stride = 1, downsample = None

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate,
                                weight_decay = 0.001,
                                momentum = 0.9)

    # Train    
    for epoch in range(num_epochs):
        print(f'--------------------Train--------------------')
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion_weighted(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del images, labels, outputs
            torch.cuda.empty_cache()
            gc.collect()

        print ('Epoch [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, num_epochs, loss.item()))
        
        # Validation
        with torch.no_grad():
            # print(f'--------------------Valid--------------------')
            correct = 0
            total = 0
            total_auc_roc = 0
            for idx, item in enumerate(valid_loader):
                images = item[0]
                labels = item[1]
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # print('Validation Set Number : {} | Accuracy : {} %'.format(idx, correct / total))

                outputs_sf_max = outputs.softmax(dim=1)
                probability = outputs_sf_max[:, 1].cpu().detach().numpy()
                labels_np = labels.cpu().detach().numpy()

                fprs, tprs, thresholds = roc_curve(labels_np, probability, pos_label=1)
                auc_roc = metrics.auc(fprs, tprs)
                total_auc_roc = total_auc_roc + auc_roc
                # print(f'fprs: {fprs} | tprs : {tprs} | auc_roc: {auc_roc}')
                
                del images, labels, outputs

        # Test : 5-fold
        total = 0
        correct = 0
        total_auc_roc = 0
        cnt = 0
        print('c_val_dataset size : ', len(c_val_dataset))
        for idx, c_val_fold in enumerate(random_split(c_val_dataset,
                                                      [int(len(c_val_dataset)/5) for idx in range(5)], 
                                                      generator=torch.Generator().manual_seed(42))
                                        ):
            
            print(f'--------------------Test : {idx} fold --------------------')
            fold_batch_size = int(len(c_val_dataset)/5)
            # fold_batch_size = batch_size
            c_val_loader = DataLoader(c_val_fold,
                                      batch_size=fold_batch_size,
                                      shuffle=True,
                                      num_workers=4)
            # Validation
            with torch.no_grad():
                for loader_idx, item in enumerate(c_val_loader):
                    images = item[0]
                    labels = item[1]
                    images = images.to(device)
                    labels = labels.to(device)
                    if torch.all(labels.bool()): 
                        print(f'{idx} valid loader is skipped!')
                        continue
                    cnt += 1
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    print('Cross Validation Fold Number : {} | Accuracy : {} %'.format(idx, correct / total))

                    outputs_sf_max = outputs.softmax(dim=1)
                    probability = outputs_sf_max[:, 1].cpu().detach().numpy()
                    labels_np = labels.cpu().detach().numpy()
                    print(f'model output probability : {probability}')

                    fprs, tprs, thresholds = roc_curve(labels_np, probability, pos_label=1)
                    auc_roc = metrics.auc(fprs, tprs)
                    total_auc_roc = total_auc_roc + auc_roc
                    if not osp.exists(f'./{timestr}'):
                        os.mkdir(f'./{timestr}')
                    fname = os.path.join(f'./{timestr}/', f'{epoch}_epoch_{idx}_fold_auc_roc.png')
                    save_auc_roc_fig(fname, label=labels_np, predict=probability)
                    print(f'Cross Validation Fold Number : {idx} | auc_roc: {auc_roc}') 

                    del images, labels, outputs

        report_loss = loss.cpu().detach().numpy()
        average_accuracy = correct / total
        avg_auc_roc = total_auc_roc / cnt
        if epoch==0:
            best_auc_roc = avg_auc_roc
        elif (best_auc_roc < avg_auc_roc):
            best_auc_roc = avg_auc_roc
            torch.save(model, f'./{timestr}/best_model_ep{epoch}.pt')
        report_doc = {'default': float(report_loss),
                      'average_accuracy': average_accuracy,
                      'avg_auc_roc': avg_auc_roc}
        nni.report_intermediate_result(report_doc)
        print(report_doc)

        if epoch == num_epochs-1:
            nni.report_final_result(best_auc_roc)
            


if __name__ == "__main__":
    main()


