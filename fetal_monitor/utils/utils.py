import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score

from matplotlib import pyplot as plt



def get_precision(_df, threshold=7):
    # TR|FP|TN|FN
    ground_positive = _df[_df['pIC50'] >= threshold]['title'].values
    predct_positive = _df[_df['rescore'] >= threshold]['title'].values

    true_positive = np.intersect1d(predct_positive, ground_positive)
    
    if len(true_positive) == 0:
        precision = 0
    else:
        # precision
        precision = len(true_positive) / (len(np.union1d(ground_positive, predct_positive)))

    return precision


def get_recall(_df, threshold=7):
    # TR|FP|TN|FN
    ground_positive = _df[_df['pIC50'] >= threshold]['title'].values
    predct_positive = _df[_df['rescore'] >= threshold]['title'].values

    true_positive = np.intersect1d(predct_positive, ground_positive)
    
    if len(true_positive) == 0:
        recall = 0
    else:
        # recall
        recall = len(true_positive) / len(ground_positive)

    return recall


def calc_auc(y_pred,y_test):
    '''
    instead of roc_auc_score
    '''
    fpr, tpr, thresholds = metrics.roc_curve(y_test,y_pred,pos_label=1)
    return metrics.auc(fpr,tpr)


def return_gold_plt(frame_data_frame):
    _corr = np.mean(frame_data_frame['gold_corr'].values)
    xx = frame_data_frame['pIC50']
    yy = frame_data_frame['Gold.PLP.Fitness_best']

    fp1 = np.polyfit(xx, yy, 1)
    f1 = np.poly1d(fp1)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(xx, 
               yy, 
               s=150, 
               lw=0.2, 
               alpha=0.5, 
               edgecolors='black', 
               color = '#1f77b4', 
               label='rescore/pIC50')
    ax.text(10,
            60,
            f'\ncorr avg. : {_corr:.3f}',
            color = '#1f77b4',
            size=10)
    ax.set_xlabel('pIC50')
    ax.set_ylabel('Fitness_best')
    ax.legend(loc='upper right')
    plt.plot(xx, f1(xx), lw=2, color='r', label='polyfit')

    return plt
