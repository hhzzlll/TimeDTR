

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch import optim

import pickle as pkl
import os
import time
import math
from typing import Callable, Optional, Union, Dict, Tuple

import warnings
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

import pickle

from data_provider.data_factory import data_provider

from exp.exp_basic import Exp_Basic
from models_diffusion import DDPM

from utils.metrics import metric, calc_quantile_CRPS
from utils.tools import visual, visual_prob, visual2D
from FEA_module.FEA import *
from FEA_module.mask import *

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--config', type=str, default='config/config_FEA.json',
                        help='JSON file for configuration')
        args_adapter = parser.parse_args()
        with open(args_adapter.config) as f:
            data = f.read()
        config = json.loads(data)
        model_config = config['FEA_config']
        self.adapter = FEA(**model_config).cuda()

    def _build_model(self):
        model_dict = {
            'DDPM': DDPM,
        }
        self.args.device = self.device
        model = model_dict[self.args.model].Model(self.args).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, shuffle_flag_train=True):

        data_set, data_loader = data_provider(self.args, flag)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        # model_optim = optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.9)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.train_epochs)

        return model_optim, lr_scheduler

    def _get_full_train_val_data(self):

        data_set, train_data = data_provider(self.args, flag='train', return_full_data=True)
        
        return train_data 

    def pretrain(self, setting):

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_data, train_loader = self._get_data(flag='train')
        if self.args.use_valset:
            vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        time_now = time.time()

        train_steps = len(train_loader)

        model_optim = optim.Adam(self.model.parameters(), lr=0.0001)

        best_train_loss = 10000000.0
        for epoch in range(self.args.pretrain_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, idx, t1, t2, max_lens) in enumerate(train_loader):

                # batch_x torch.Size([bsz, seq_len, fea_dim])
                # batch_y torch.Size([bsz, label_len+pred_len, fea_dim])

                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = batch_y

                loss = self.model.pretrain_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                train_loss.append(loss.item())

                loss.backward()

                model_optim.step()

            print("PreTraining Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            print("PreTraining Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} ".format(epoch + 1, train_steps, train_loss))

            if train_loss < best_train_loss:
                print("-------------------------")
                best_train_loss = train_loss
                torch.save(self.model.dlinear_model.state_dict(), path + '/' + 'pretrain_model_checkpoint.pth')

    def train(self, setting):

        if self.args.model in ['depts']:
            full_train_seqs = self._get_full_train_val_data()
            # print("full_train_seqs", np.shape(full_train_seqs)) # (18412, 321)
            # print("full_val_seqs", np.shape(full_val_seqs)) # (2728, 321)
            fast_len = min([np.shape(full_train_seqs)[0], 2048])
            full_train_seqs = full_train_seqs[-fast_len:]
            fftwarmlen = int(np.shape(full_train_seqs)[0]*0.5)
            self.model.initialize(full_train_seqs, fftwarmlen)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
         
        if self.args.stage_mode == "TWO":
            best_model_path = path + '/' + 'pretrain_model_checkpoint.pth'
            self.model.dlinear_model.load_state_dict(torch.load(best_model_path))
            print("Successfully loading pretrained model!")

        train_data, train_loader = self._get_data(flag='train')
        if self.args.use_valset:
            vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        time_now = time.time()

        train_steps = len(train_loader)

        model_optim, lr_scheduler = self._select_optimizer()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        best_train_loss = 10000000.0
        training_process = {}
        training_process["train_loss"] = []
        training_process["val_loss"] = []

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            self.adapter.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, idx, t1, t2, max_lens) in enumerate(train_loader):

                # batch_x torch.Size([bsz, seq_len, fea_dim])
                # batch_y torch.Size([bsz, label_len+pred_len, fea_dim])

                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = batch_y

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        loss = self.model.train_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'SDSB' in self.args.model:
                        loss = self.model.train_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark, epoch=epoch)
                    else:
                        loss = self.model.train_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark, self.adapter)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
                    scaler.update()
                else:
                    loss.backward()
                    # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10, norm_type=2)
                    model_optim.step()
                    
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            training_process["train_loss"].append(train_loss)

            if epoch % 1 == 0:

                val_loss = self.val(setting, vali_loader)
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Val Loss: {3:.7f} ".format(epoch + 1, train_steps, train_loss, val_loss))
                training_process["val_loss"].append(val_loss)

                if val_loss < best_train_loss:
                    print("-------------------------")
                    best_train_loss = val_loss
                    best_model_path = path + '/' + 'checkpoint.pth'
                    torch.save(self.model.state_dict(), path + '/' + 'Model_checkpoint.pth')
                    torch.save(self.adapter.state_dict(), path + '/' + 'adapter_checkpoint.pth')

            lr_scheduler.step()

        best_model_path = path + '/' + 'Model_checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        best_adapter_path = path + '/' + 'adapter_checkpoint.pth'
        self.adapter.load_state_dict(torch.load(best_adapter_path))

        f=open(path + '/' + 'losses.pkl','wb')
        pkl.dump(training_process,f)
        f.close()

        return self.model

    def val(self, setting, vali_loader):

        test_loader = vali_loader

        inps = []    
        preds = []
        trues = []

        self.model.eval()
        self.adapter.eval()

        # with torch.no_grad():

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, idx, t1, t2, max_lens) in enumerate(test_loader):

            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            dec_inp = batch_y

            outputs, batch_x, batch_y, mean, label_part = self.model.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark, adapter=self.adapter, sample_times=1)

            if len(np.shape(outputs)) == 4:
                outputs = outputs.mean(dim=1)

            pred = outputs.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()

            preds.append(pred)
            trues.append(true)
            inps.append(batch_x.detach().cpu().numpy())

            # if self.args.dataset_name not in ["Exchange"]:
            if i > 5:
                break

        inps = np.array(inps)
        preds = np.array(preds)
        trues = np.array(trues)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe, corr = metric(preds, trues)
        
        self.model.train()
        self.adapter.train()

        return mse

    def test(self, setting, mode="test"):

        test_data, test_loader = test_data, test_loader = self._get_data(flag=mode)

        if self.args.model not in ['ARIMA']:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'Model_checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

            best_adapter_path = path + '/' + 'adapter_checkpoint.pth'
            self.adapter.load_state_dict(torch.load(best_adapter_path))

            print("Successfully loading trained model!")

        inps = []   
        preds = []
        all_generated_samples = []
        trues = []
        time_stamps = []
        
        return_mean = []
        return_label = []

        folder_path = os.path.join(self.args.checkpoints, setting)

        self.model.eval()
        self.adapter.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, idx, t1, t2, max_lens) in enumerate(test_loader):
                print(i)
                # if i % 20 == 0 and i < 100 and (i!=20):
                if True: # i == 20:

                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    # print(np.shape(batch_x), np.shape(idx), np.shape(t1))
                    # torch.Size([32, 96, 321]) torch.Size([32]) torch.Size([32, 96])

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    dec_inp = batch_y
                    

                    start_time = time.time()
                    sample_times = self.args.sample_times 
                    outputs, batch_x, batch_y, mean, label_part = self.model.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark, adapter=self.adapter, sample_times=sample_times)
                    end_time = time.time()
                    elapsed_time_ms = (end_time - start_time) * 1000 / np.shape(batch_x)[0]

                    if i < 5:
                        print(f"Elapsed time: {elapsed_time_ms:.2f} ms")

                    pred = outputs.detach().cpu().numpy()
                    true = batch_y.detach().cpu().numpy()
                    return_mean_i = mean
                    return_label_i = label_part

                    # (32, 10, 96, 1) 
                    # (B,nsample,L,K)

                    if len(np.shape(pred)) == 4:
                        preds.append(pred.mean(axis=1))
                        if self.args.sample_times > 1:
                            all_generated_samples.append(pred)
                    else:
                        preds.append(pred)
                    trues.append(true)

                    if return_mean_i is not None:
                        return_mean.append(return_mean_i.detach().cpu().numpy())
                    if return_label_i is not None:
                        return_label.append(return_label_i.detach().cpu().numpy())

                    time_stamps.append(t2[:, -self.args.pred_len:])

                    if self.args.model not in ['ARIMA']:
                        if self.args.out_figures > 0 and i % 10 == 0 and i < 100:

                            if self.args.dataset_name == "system_KS":
                                input = batch_x.detach().cpu().numpy()
                                if len(np.shape(pred)) == 4:
                                    pred = pred.mean(axis=1)
                                his = input[0]
                                gt = true[0]
                                pd = pred[0]
                                his = test_data.inverse_transform(his)[-96:, :]
                                gt = test_data.inverse_transform(gt)
                                pd = test_data.inverse_transform(pd)
                                visual2D(his, gt, pd, os.path.join(folder_path, mode + str(i) + '.png'))
                            else:
                                id_worst = -1 # 182 # -1
                                input = batch_x.detach().cpu().numpy()
                                # his = input[0, -self.args.seq_len:, id_worst]
                                his = input[0, -336:, id_worst]
                                gt = true[0, :, id_worst]
                                if return_mean_i is not None:
                                    return_mean_i = return_mean_i.detach().cpu().numpy()[0, :, id_worst]
                                if return_label_i is not None:
                                    return_label_i = return_label_i.detach().cpu().numpy()[0, :, id_worst]

                                if self.args.sample_times > 1:
                                    pd = pred[0][:, :, id_worst]
                                    prob_pd = all_generated_samples[-1][0][:, :, id_worst]
                                    # pred: (32, 3, 96, 1)
                                    visual_prob(self.args, his, gt, pd, name=os.path.join(folder_path, mode + str(i) + '.png'), mean_pred=return_mean_i, label_part=return_label_i, prob_pd=prob_pd)
                                else:
                                    # print(">>>>>>>>>>>", np.shape(pred))
                                    pd = pred[0][0, :, id_worst]
                                    visual(self.args, his, gt, pd, name=os.path.join(folder_path, mode + str(i) + '.png'), mean_pred=return_mean_i, label_part=return_label_i)

                        # pickle.dump(pd, open(os.path.join(folder_path,'result_{}.pkl'.format(i)), 'wb'))

                    # if i > 10:
                    #     break

        inps = np.array(inps)
        preds = np.array(preds)
        trues = np.array(trues)
        id_worst = None

        if self.args.features == 'M' and self.args.vis_MTS_analysis:
            
            N, B, L, D = np.shape(preds)
            VIS_P = preds.reshape((N*B, L, D))        
            VIS_T = trues.reshape((N*B, L, D))   
            # print(">>>", np.shape(VIS_P), np.shape(VIS_T))

            res = np.mean((VIS_P - VIS_T) ** 2, axis=1)
            res = np.mean(res, axis=0)
            # print(">>>", np.shape(res))

            print("id_worst", np.argmax(res))
            id_worst = np.argmax(res)

            ind = np.argpartition(res, -5)[-5:]
            top5 = res[ind]
            print("top5", ind) # max

            plt.figure(figsize=(12,5))
            plt.bar(range(self.args.num_vars),res,align = "center",color = "steelblue",alpha = 0.6)
            plt.ylabel("MSE")
            plt.savefig(os.path.join(folder_path, 'MTS_errors.png'))
            
            plt.figure(figsize=(10,5))
            plt.hist(res, bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
            plt.xlabel("mse")
            plt.ylabel("frequency")
            plt.savefig(os.path.join(folder_path, 'MTS_errors_hist.png'))
            
        # print(">>>", np.shape(preds), np.shape(trues))
        # >>> (158, 32, 192, 321) (158, 32, 192, 321)

        if self.args.sample_times > 1:
            all_generated_samples = np.array(all_generated_samples)

        # print("preds", np.shape(preds))
        preds = preds.reshape(-1, trues.shape[-2], trues.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        if self.args.sample_times > 1:
            all_generated_samples = all_generated_samples.reshape(-1, self.args.sample_times , trues.shape[-2], trues.shape[-1])
            # print('test shape:', preds.shape, trues.shape, all_generated_samples.shape)
            # (224, 96, 1) (224, 96, 1) (224, 10, 96, 1)

            crps = calc_quantile_CRPS(all_generated_samples, trues)

        # result save
        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, corr = metric(preds, trues)

        if self.args.sample_times > 1:
            print('mse|mae|crps|rmse|mape|mspe|corr')
            print(mse, mae, crps, rmse, mape, mspe, corr)

        else:
            print('mse|mae|rmse|mape|mspe|corr')
            print(mse, mae, rmse, mape, mspe, corr)
        # print('mse:{}, mae:{}'.format(mse, mae))
        # print('rmse:{}, mape:{}, mspe:{}, corr:{}'.format(rmse, mape, mspe, corr))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}, corr:{}'.format(mse, mae, rmse, mape, mspe, corr))
        f.write('\n')
        f.write('\n')
        f.close()
        return mae, mse, rmse, mape, mspe, corr

