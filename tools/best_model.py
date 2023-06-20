import os
import os.path as osp
    
from glob import glob
import torch
import torch.nn as nn
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
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import KFold

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

CROSSVAL = [11977062, 12144707, 12206951, 11632775]
CROSSVAL_BASKET = list()

resize_scala = 224
num_classes = 2

def main(model_path):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet(ResidualBlock, [3, 2, 6, 3], num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    for patient_id_path in glob(f'/home/sangwon/sources/resnet/tools/data/*Data*0313*/*'):
        patient_id = int(osp.basename(patient_id_path))
        if patient_id in CROSSVAL:
            _dataset = ImageFolder(root=patient_id_path,
                                   transform=transforms.Compose([transforms.Resize([resize_scala, resize_scala]),
                                                                 transforms.ToTensor(),
                                                                ]))
        CROSSVAL_BASKET.append(_dataset)

        if (len(CROSSVAL_BASKET) == len(CROSSVAL)):
            c_val_dataset = ConcatDataset(CROSSVAL_BASKET)
        else:
            raise Exception(f'There should be patient ids omitted!')
            
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



if __name__ == "__main__":
    main()
