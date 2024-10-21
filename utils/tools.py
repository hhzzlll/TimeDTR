
import numpy as np
import torch
import matplotlib.pyplot as plt

plt.switch_backend('agg')

from torch import optim
import pickle


def visual(args, history, true, preds=None, mean_pred=None, label_part=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure(figsize=(8,5))
    ind_his = list(np.arange(0,len(history)))
    ind_out = list(np.arange(len(history), len(history)+len(true)))
    if label_part is not None:
        label_out = list(np.arange(len(history)-len(label_part), len(history)))
    plt.plot(ind_his, history, '-', label='History', c='#000000', linewidth=1)
    plt.plot([ind_his[-1], ind_his[-1]+1], [history[-1], true[0]], '-', c='#000000', linewidth=1)
    
    plt.plot(ind_out, true, '-', label='GroundTruth', c='b', linewidth=1) # #999999
    if mean_pred is not None:
        plt.plot(ind_out, mean_pred, '-', label='Pred-Trend', c='gray', linewidth=1)
    if preds is not None:
        # print(np.shape(ind_out), np.shape(preds))
        plt.plot(ind_out, preds, '-', label='Prediction', c='r', linewidth=1)  # #FFB733    
    if label_part is not None:
        plt.plot(label_out, label_part, '-', label='Pred-Label', c='pink', linewidth=1)

    plt.legend()
    plt.tight_layout()
    plt.savefig(name, bbox_inches='tight')

    f = open(name[:-4]+'.pkl', "wb")
    pickle.dump(preds, f)
    f.close()

    f = open(name[:-4]+'_ground_truth.pkl', "wb")
    pickle.dump(true, f)
    f.close()

    f = open(name[:-4]+'_history.pkl', "wb")
    pickle.dump(history, f)
    f.close()

def visual_prob(args, history, true, preds=None, mean_pred=None, label_part=None, name='./pic/test.pdf', prob_pd=None):
    """
    Results visualization
    """
    plt.figure(figsize=(8,5))
    ind_his = list(np.arange(0,len(history)))
    ind_out = list(np.arange(len(history), len(history)+len(true)))
    if label_part is not None:
        label_out = list(np.arange(len(history)-len(label_part), len(history)))
    plt.plot(ind_his, history, '-', label='History', c='#000000', linewidth=1)
    plt.plot([ind_his[-1], ind_his[-1]+1], [history[-1], true[0]], '-', c='#000000', linewidth=1)
    
    plt.plot(ind_out, true, '-', label='GroundTruth', c='b', linewidth=1) # #999999
    if mean_pred is not None:
        plt.plot(ind_out, mean_pred, '-', label='Pred-Trend', c='gray', linewidth=1)
    if preds is not None:
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>...", np.shape(preds)) # (10, 96)
        mean = np.mean(preds, axis=0).reshape(-1, 1)
        std = np.std(preds, axis=0).reshape(-1, 1)
        # ind_out = ind_out.reshape(-1, 1)
        
        if args.sample_times > 1:
            ub = mean + std
            lb = mean - std
            new_ind_out = np.expand_dims(np.array(ind_out), axis=1)[:,0]
            print(np.shape(new_ind_out), np.shape(ub), np.shape(lb), np.shape(ind_out), np.shape(mean), np.shape(std))
            plt.fill_between(new_ind_out, ub[:,0], lb[:,0], color="#b9cfe7", edgecolor=None)
        # plt.fill_between(ind_out, mean + std, mean - std, facecolor="gray")
        plt.plot(ind_out, mean, '-', label='Prediction', c='r', linewidth=1)  # #FFB733    
    if label_part is not None:
        plt.plot(label_out, label_part, '-', label='Pred-Label', c='pink', linewidth=1)

    plt.legend()
    plt.tight_layout()
    print(name)
    # plt.show()
    plt.savefig(name, bbox_inches='tight') 
    
    f = open(name[:-4]+'.pkl', "wb")
    pickle.dump(preds, f)
    f.close()

    f = open(name[:-4]+'_ground_truth.pkl', "wb")
    pickle.dump(true, f)
    f.close()

    f = open(name[:-4]+'_history.pkl', "wb")
    pickle.dump(history, f)
    f.close()


def visual2D(history, true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    # print(np.shape(history), np.shape(true), np.shape(preds))

    gtrue = np.concatenate([history, true], axis=0)
    preds = np.concatenate([history, preds], axis=0)

    # print(np.shape(gtrue), np.shape(preds))

    plt.figure(figsize=(14,5))
    cmap = 'jet'
    aspect = 1
    plt.subplot(211)
    plt.imshow(gtrue.T, cmap=cmap, aspect=aspect, vmin=-0.3, vmax=0.3)
    plt.subplot(212)
    plt.imshow(preds.T, cmap=cmap, aspect=aspect, vmin=-0.3, vmax=0.3)
    plt.tight_layout()
    plt.savefig(name)

    
