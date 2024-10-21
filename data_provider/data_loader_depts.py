

import os
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.data_utils import group_values
from datetime import datetime

from sklearn.preprocessing import StandardScaler
import pickle as pkl

import warnings
warnings.filterwarnings('ignore')


class Dataset_Caiso(Dataset):
    def __init__(self, args, root_path, flag='train'):

        set_split = {"last18months":"2020-01-01 00", "last15months":"2020-04-01 00",
             "last12months":"2020-07-01 00", "last9months":"2020-10-01 00"}

        self.flag = flag
        self.args = args

        self.insample_size = args.seq_len
        self.label_len = args.label_len
        self.outsample_size = args.pred_len

        DATA_DIR = os.path.join(root_path, 'caiso_20130101_20210630.csv')
        data = pd.read_csv(DATA_DIR)

        data['Date'] = data['Date'].astype('datetime64')
        names = ['PGE','SCE','SDGE','VEA','CA ISO','PACE','PACW','NEVP','AZPS','PSEI']
        ids = np.arange(len(names))
        df_all = pd.DataFrame(pd.date_range('20130101','20210630',freq='H')[:-1], columns=['Date'])
        for name in names:
            current_df = data[data['zone'] == name].drop_duplicates(subset='Date', keep='last').rename(columns={'load':name}).drop(columns=['zone'])
            df_all = df_all.merge(current_df, on='Date', how='outer')

        # set index
        df_all = df_all.set_index('Date')
        values = df_all.fillna(0).values.T
        dates = np.array([str(x)[:13] for x in df_all.index.tolist()])

        self.ids = ids

        # NORMALIZATION
        self.scaler = StandardScaler()
        self.values = values
        self.dates = dates

        print(">>> values", np.shape(self.values))
        # print(self.ids)
        # (10, 74472)
        # [0 1 2 3 4 5 6 7 8 9]
        """
        from statsmodels.tsa.stattools import adfuller
        data = self.values.T
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(np.shape(data))
        print(self.args.dataset_name)
        result = adfuller(data[:,-1])
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        """
        val_cut_date = set_split["last18months"]
        val_date = datetime.strptime(val_cut_date, '%Y-%m-%d %H')
        test_cut_date = set_split["last9months"]
        test_date = datetime.strptime(test_cut_date, '%Y-%m-%d %H')

        left_indices = []
        mid_indices = []
        right_indices = []
        for i, p in enumerate(self.dates):
            record_date = datetime.strptime(p, '%Y-%m-%d %H')
            if record_date < val_date:
                left_indices.append(i)
            else:
                if record_date < test_date:
                    mid_indices.append(i)
                else:
                    right_indices.append(i)

        print("train/val/test:{}/{}/{}".format(len(left_indices), len(mid_indices), len(right_indices)))

        if flag == 'train':
            self.indices = left_indices
            self.extracted_values = self.values[:, left_indices]
            self.extracted_dates = self.dates[left_indices]

            for i in range(len(self.extracted_values)):
                self.scaler = StandardScaler()
                temp = np.expand_dims(np.asarray(self.extracted_values[i]), axis=1)
                self.scaler.fit(temp)
                self.extracted_values[i] = self.scaler.transform(temp).squeeze(1)

            # self.scaler.fit(self.extracted_values)
            # self.extracted_values = self.scaler.transform(self.extracted_values)

        else:
            # self.extracted_values = self.values[:, right_indices]
            self.train_values = self.values[:, left_indices]
            if flag == 'val':
                self.indices = mid_indices
                self.test_values = self.values[:, mid_indices]
            else:
                self.indices = right_indices
                self.test_values = self.values[:, right_indices]

            for i in range(len(self.train_values)):
                self.scaler = StandardScaler()
                temp = np.expand_dims(np.asarray(self.train_values[i]), axis=1)
                temp_test = np.expand_dims(np.asarray(self.test_values[i]), axis=1)
                self.scaler.fit(temp)
                self.test_values[i] = self.scaler.transform(temp_test).squeeze(1)
            # self.scaler.fit(self.values[:, left_indices])
            # self.extracted_values = self.scaler.transform(self.extracted_values)
            self.extracted_values = self.test_values
            self.extracted_dates = self.dates[right_indices]

        self.lens = [len(self.extracted_values[i]) for i in range(np.shape(self.extracted_values)[0])]
        
        self.values = self.extracted_values

        self.dim_datetime_feats = -1

    def __getitem__(self, index):

        while True:
            insample = np.zeros((self.insample_size, 1))
            outsample = np.zeros((self.outsample_size+self.label_len, 1))
            
            sampled_index = np.random.randint(np.shape(self.extracted_values)[0])
            
            sampled_timeseries = self.extracted_values[sampled_index]

            cut_point = np.random.randint(low=self.insample_size, high=len(sampled_timeseries)-self.outsample_size, size=1)[0]

            insample_window = sampled_timeseries[cut_point - self.insample_size:cut_point]
            insample = np.expand_dims(insample_window, 1)

            outsample_window = sampled_timeseries[(cut_point-self.label_len):(cut_point+self.outsample_size)]
            outsample[:len(outsample_window)] = np.expand_dims(outsample_window, 1)

            # =================================================================================
            # global time steps
            idx = sampled_index
            timestamps1 = np.asarray(list(range(cut_point - self.insample_size, cut_point)))
            timestamps2 = np.asarray(list(range((cut_point-self.label_len), (cut_point + self.outsample_size))))

            if np.max(insample_window) != np.min(insample_window):
                break

        return insample, outsample, np.zeros([np.shape(insample)[0], 1]), np.zeros([np.shape(outsample)[0], 1]), idx, timestamps1, timestamps2, len(sampled_timeseries)

    def __len__(self):

        if self.flag == 'train':
            return 5000
        else:
            return int(len(self.extracted_values)*((np.min(self.lens)+np.max(self.lens))/2))


class Dataset_Production(Dataset):
    def __init__(self, args, root_path, flag='train'):

        set_split = {"last12months":"2020-01-01 00","last9months":"2020-04-01 00",
               "last6months":"2020-07-01 00", "last3months":"2020-10-01 00"}

        self.flag = flag
        self.args = args

        self.insample_size = args.seq_len
        self.label_len = args.label_len
        self.outsample_size = args.pred_len

        DATA_PATH = os.path.join(root_path, 'production.csv')
        data = pd.read_csv(DATA_PATH, parse_dates=['Time'])
        data = data.set_index('Time')
        ids = np.arange(data.shape[1])
        values = data.fillna(0).values.T
        dates = np.array([str(x)[:13] for x in data.index.tolist()])

        self.ids = ids
        self.values = values
        self.dates = dates

        # cut_date = set_split["last12months"]
        # date = datetime.strptime(cut_date, '%Y-%m-%d %H')
        print(">>> values", np.shape(self.values))
        """
        from statsmodels.tsa.stattools import adfuller
        data = self.values.T
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(np.shape(data))
        print(self.args.dataset_name)
        result = adfuller(data[:,-1])
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        """
        val_cut_date = set_split["last9months"]
        val_date = datetime.strptime(val_cut_date, '%Y-%m-%d %H')
        test_cut_date = set_split["last3months"]
        test_date = datetime.strptime(test_cut_date, '%Y-%m-%d %H')

        left_indices = []
        mid_indices = []
        right_indices = []
        for i, p in enumerate(self.dates):
            record_date = datetime.strptime(p, '%Y-%m-%d %H')
       
            if record_date < val_date:
                left_indices.append(i)
            else:
                if record_date < test_date:
                    mid_indices.append(i)
                else:
                    right_indices.append(i)

        self.scaler = StandardScaler()

        print("train/val/test:{}/{}/{}".format(len(left_indices), len(mid_indices), len(right_indices)))

        if flag == 'train':
            self.extracted_values = self.values[:, left_indices]
            self.extracted_dates = self.dates[left_indices]

            for i in range(len(self.extracted_values)):
                self.scaler = StandardScaler()
                temp = np.expand_dims(np.asarray(self.extracted_values[i]), axis=1)
                self.scaler.fit(temp)
                self.extracted_values[i] = self.scaler.transform(temp).squeeze(1)

        else:
            # self.extracted_values = self.values[:, right_indices]

            # self.scaler.fit(self.values[:, left_indices])
            # self.extracted_values = self.scaler.transform(self.extracted_values)

            self.train_values = self.values[:, left_indices]
            if flag == 'val':
                self.indices = mid_indices
                self.test_values = self.values[:, mid_indices]
            else:
                self.indices = right_indices
                self.test_values = self.values[:, right_indices]

            for i in range(len(self.train_values)):
                self.scaler = StandardScaler()
                temp = np.expand_dims(np.asarray(self.train_values[i]), axis=1)
                temp_test = np.expand_dims(np.asarray(self.test_values[i]), axis=1)
                self.scaler.fit(temp)
                self.test_values[i] = self.scaler.transform(temp_test).squeeze(1)

            self.extracted_values = self.test_values
            self.extracted_dates = self.dates[right_indices]

        self.lens = [len(self.extracted_values[i]) for i in range(np.shape(self.extracted_values)[0])]
        self.values = self.extracted_values

        self.dim_datetime_feats = -1

    def __getitem__(self, index):

        while True:
            sampled_index = np.random.randint(len(self.extracted_values))
            sampled_timeseries = self.extracted_values[sampled_index]

            cut_point = np.random.randint(low=self.insample_size, high=len(sampled_timeseries)-self.outsample_size, size=1)[0]

            insample_window = sampled_timeseries[cut_point - self.insample_size:cut_point]
            insample = np.expand_dims(insample_window, 1)
            
            outsample_window = sampled_timeseries[(cut_point-self.label_len):(cut_point + self.outsample_size)]
            outsample = np.expand_dims(outsample_window, 1)
            
            # =================================================================================
            # global time steps
            idx = sampled_index
            timestamps1 = np.asarray(list(range(cut_point - self.insample_size, cut_point)))
            timestamps2 = np.asarray(list(range((cut_point-self.label_len), (cut_point + self.outsample_size))))

            if np.max(insample_window) != np.min(insample_window):
                break

        return insample, outsample, np.zeros([np.shape(insample)[0], 1]), np.zeros([np.shape(outsample)[0], 1]), idx, timestamps1, timestamps2, len(sampled_timeseries)

    def __len__(self):

        # return int(len(self.extracted_values)*((np.min(self.lens)+np.max(self.lens))/2))
        if self.flag == 'train':
            return 5000
        else:
            return int(len(self.extracted_values)*((np.min(self.lens)+np.max(self.lens))/2))


class Dataset_Caiso_M(Dataset):
    def __init__(self, args, root_path, flag='train'):

        set_split = {"last18months":"2020-01-01 00", "last15months":"2020-04-01 00",
             "last12months":"2020-07-01 00", "last9months":"2020-10-01 00"}

        self.flag = flag
        self.args = args

        self.insample_size = args.seq_len
        self.label_len = args.label_len
        self.outsample_size = args.pred_len

        DATA_DIR = os.path.join(root_path, 'caiso_20130101_20210630.csv')
        data = pd.read_csv(DATA_DIR)

        data['Date'] = data['Date'].astype('datetime64')
        names = ['PGE','SCE','SDGE','VEA','CA ISO','PACE','PACW','NEVP','AZPS','PSEI']
        ids = np.arange(len(names))
        df_all = pd.DataFrame(pd.date_range('20130101','20210630',freq='H')[:-1], columns=['Date'])
        for name in names:
            current_df = data[data['zone'] == name].drop_duplicates(subset='Date', keep='last').rename(columns={'load':name}).drop(columns=['zone'])
            df_all = df_all.merge(current_df, on='Date', how='outer')

        # set index
        df_all = df_all.set_index('Date')
        values = df_all.fillna(0).values.T
        dates = np.array([str(x)[:13] for x in df_all.index.tolist()])

        self.ids = ids

        # NORMALIZATION
        self.scaler = StandardScaler()
        self.values = values
        self.dates = dates

        print(">>> values", np.shape(self.values))
        # print(self.ids)
        # (10, 74472)
        # [0 1 2 3 4 5 6 7 8 9]

        val_cut_date = set_split["last18months"]
        val_date = datetime.strptime(val_cut_date, '%Y-%m-%d %H')
        test_cut_date = set_split["last9months"]
        test_date = datetime.strptime(test_cut_date, '%Y-%m-%d %H')

        left_indices = []
        mid_indices = []
        right_indices = []
        for i, p in enumerate(self.dates):
            record_date = datetime.strptime(p, '%Y-%m-%d %H')
            if record_date < val_date:
                left_indices.append(i)
            else:
                if record_date < test_date:
                    mid_indices.append(i)
                else:
                    right_indices.append(i)

        print("train/val/test:{}/{}/{}".format(len(left_indices), len(mid_indices), len(right_indices)))

        if flag == 'train':
            self.indices = left_indices
            self.extracted_values = self.values[:, left_indices]
            self.extracted_dates = self.dates[left_indices]

            for i in range(len(self.extracted_values)):
                self.scaler = StandardScaler()
                temp = np.expand_dims(np.asarray(self.extracted_values[i]), axis=1)
                self.scaler.fit(temp)
                self.extracted_values[i] = self.scaler.transform(temp).squeeze(1)

            # self.scaler.fit(self.extracted_values)
            # self.extracted_values = self.scaler.transform(self.extracted_values)

        else:
            # self.extracted_values = self.values[:, right_indices]
            self.train_values = self.values[:, left_indices]
            if flag == 'val':
                self.indices = mid_indices
                self.test_values = self.values[:, mid_indices]
            else:
                self.indices = right_indices
                self.test_values = self.values[:, right_indices]

            for i in range(len(self.train_values)):
                self.scaler = StandardScaler()
                temp = np.expand_dims(np.asarray(self.train_values[i]), axis=1)
                temp_test = np.expand_dims(np.asarray(self.test_values[i]), axis=1)
                self.scaler.fit(temp)
                self.test_values[i] = self.scaler.transform(temp_test).squeeze(1)
            # self.scaler.fit(self.values[:, left_indices])
            # self.extracted_values = self.scaler.transform(self.extracted_values)
            self.extracted_values = self.test_values
            self.extracted_dates = self.dates[right_indices]

        self.lens = [len(self.extracted_values[i]) for i in range(np.shape(self.extracted_values)[0])]

        self.values = self.extracted_values.T

        # print(">>>>>>>>>>>>>>>>>>>>>>", self.flag, np.shape(self.values))

        self.dim_datetime_feats = -1

        self.set_len = np.shape(self.values)[0]-self.outsample_size-self.insample_size

    def __getitem__(self, index):

        sampled_timeseries = self.values

        cut_point = index

        insample = sampled_timeseries[cut_point:(cut_point+self.insample_size)]
        outsample = sampled_timeseries[(cut_point+self.insample_size-self.label_len):(cut_point+self.insample_size+self.outsample_size)]

        # print(index, cut_point, self.set_len, np.shape(insample), np.shape(outsample))

        # =================================================================================
        # global time steps
        idx = 0
        timestamps1 = np.asarray(list(range(cut_point,(cut_point+self.insample_size))))
        timestamps2 = np.asarray(list(range((cut_point+self.insample_size-self.label_len),(cut_point+self.insample_size+self.outsample_size))))

        return insample, outsample, np.zeros([np.shape(insample)[0], 1]), np.zeros([np.shape(outsample)[0], 1]), idx, timestamps1, timestamps2, len(sampled_timeseries)

    def __len__(self):

        return self.set_len


class Dataset_Production_M(Dataset):
    def __init__(self, args, root_path, flag='train'):

        set_split = {"last12months":"2020-01-01 00","last9months":"2020-04-01 00",
               "last6months":"2020-07-01 00", "last3months":"2020-10-01 00"}

        self.flag = flag
        self.args = args

        self.insample_size = args.seq_len
        self.label_len = args.label_len
        self.outsample_size = args.pred_len

        DATA_PATH = os.path.join(root_path, 'production.csv')
        data = pd.read_csv(DATA_PATH, parse_dates=['Time'])
        data = data.set_index('Time')
        ids = np.arange(data.shape[1])
        values = data.fillna(0).values.T
        dates = np.array([str(x)[:13] for x in data.index.tolist()])

        self.ids = ids
        self.values = values
        self.dates = dates

        # cut_date = set_split["last12months"]
        # date = datetime.strptime(cut_date, '%Y-%m-%d %H')
        print(">>> values", np.shape(self.values))

        val_cut_date = set_split["last9months"]
        val_date = datetime.strptime(val_cut_date, '%Y-%m-%d %H')
        test_cut_date = set_split["last3months"]
        test_date = datetime.strptime(test_cut_date, '%Y-%m-%d %H')

        left_indices = []
        mid_indices = []
        right_indices = []
        for i, p in enumerate(self.dates):
            record_date = datetime.strptime(p, '%Y-%m-%d %H')
       
            if record_date < val_date:
                left_indices.append(i)
            else:
                if record_date < test_date:
                    mid_indices.append(i)
                else:
                    right_indices.append(i)

        self.scaler = StandardScaler()

        print("train/val/test:{}/{}/{}".format(len(left_indices), len(mid_indices), len(right_indices)))

        if flag == 'train':
            self.extracted_values = self.values[:, left_indices]
            self.extracted_dates = self.dates[left_indices]

            for i in range(len(self.extracted_values)):
                self.scaler = StandardScaler()
                temp = np.expand_dims(np.asarray(self.extracted_values[i]), axis=1)
                self.scaler.fit(temp)
                self.extracted_values[i] = self.scaler.transform(temp).squeeze(1)

        else:
            # self.extracted_values = self.values[:, right_indices]

            # self.scaler.fit(self.values[:, left_indices])
            # self.extracted_values = self.scaler.transform(self.extracted_values)

            self.train_values = self.values[:, left_indices]
            if flag == 'val':
                self.indices = mid_indices
                self.test_values = self.values[:, mid_indices]
            else:
                self.indices = right_indices
                self.test_values = self.values[:, right_indices]

            for i in range(len(self.train_values)):
                self.scaler = StandardScaler()
                temp = np.expand_dims(np.asarray(self.train_values[i]), axis=1)
                temp_test = np.expand_dims(np.asarray(self.test_values[i]), axis=1)
                self.scaler.fit(temp)
                self.test_values[i] = self.scaler.transform(temp_test).squeeze(1)

            self.extracted_values = self.test_values
            self.extracted_dates = self.dates[right_indices]

        self.lens = [len(self.extracted_values[i]) for i in range(np.shape(self.extracted_values)[0])]
        self.values = self.extracted_values.T

        # print(">>>>>>>>>>>>>>>>>>>>>>", self.flag, np.shape(self.values))

        self.dim_datetime_feats = -1

        self.set_len = np.shape(self.values)[0]-self.outsample_size-self.insample_size

    def __getitem__(self, index):

        sampled_timeseries = self.values

        cut_point = index

        insample = sampled_timeseries[cut_point:(cut_point+self.insample_size)]
        outsample = sampled_timeseries[(cut_point+self.insample_size-self.label_len):(cut_point+self.insample_size+self.outsample_size)]

        # print(index, cut_point, self.set_len, np.shape(insample), np.shape(outsample))

        # =================================================================================
        # global time steps
        idx = 0
        timestamps1 = np.asarray(list(range(cut_point,(cut_point+self.insample_size))))
        timestamps2 = np.asarray(list(range((cut_point+self.insample_size-self.label_len),(cut_point+self.insample_size+self.outsample_size))))

        return insample, outsample, np.zeros([np.shape(insample)[0], 1]), np.zeros([np.shape(outsample)[0], 1]), idx, timestamps1, timestamps2, len(sampled_timeseries)

    def __len__(self):

        return self.set_len




class Dataset_Synthetic(Dataset):
    def __init__(self, args, root_path, flag='train'):
        self.args = args
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len

        synthetic_mode_map = {'L':0, 'Q':1, 'C':2, 'LT':3, 'QT':4, 'CT':5}
        mode = synthetic_mode_map[self.args.synthetic_mode]
        
        self.insample_size = args.seq_len
        self.outsample_size = args.pred_len

        DATA_PATH = os.path.join(root_path, 'synthetic_data.pkl')
        with open(DATA_PATH,'rb') as f:
            values = pkl.load(f)
            # print(x.shape)
            # (10000, 6)

        self.values = data = values[:, mode]
        self.num_samples = np.shape(values)[0] 
        num_train = int(self.num_samples * 0.7)
        num_test = int(self.num_samples * 0.2)
        num_vali = self.num_samples - num_train - num_test

        border1s = [0, num_train - self.seq_len, self.num_samples - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, self.num_samples]
        
        self.scaler = StandardScaler()
        # print("self.data_x", np.shape(self.data_x))
        # self.data_x (7000,)

        if flag == 'train':
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            self.data_x = data[border1:border2]
        
            self.scaler = StandardScaler()
            temp = np.expand_dims(np.asarray(self.data_x), axis=1)
            self.scaler.fit(temp)
            self.data_x = self.scaler.transform(temp) # .squeeze(1)

        else:
            border1 = border1s[0]
            border2 = border2s[0]
            train_data = data[border1:border2]
        
            self.scaler = StandardScaler()
            temp = np.expand_dims(np.asarray(train_data), axis=1)
            self.scaler.fit(temp)

            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            self.data_x = data[border1:border2]
            test_temp = np.expand_dims(np.asarray(self.data_x), axis=1)
            self.data_x = self.scaler.transform(test_temp)# .squeeze(1)

        self.data_y = self.data_x
        self.values = self.data_x

        self.dim_datetime_feats = -1

    def __getitem__(self, index):

        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        idx = 0
        timestamps1 = np.asarray(list(range(s_begin, s_end)))
        timestamps2 = np.asarray(list(range(r_begin, r_end)))

        return seq_x, seq_y, np.zeros([np.shape(seq_x)[0], 1]), np.zeros([np.shape(seq_y)[0], 1]), idx, timestamps1, timestamps2, len(self.data_x)

    def __len__(self):

        return len(self.data_x) - self.seq_len - self.pred_len + 1

