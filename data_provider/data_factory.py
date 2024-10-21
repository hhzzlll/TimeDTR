
from data_provider.data_loader_depts import Dataset_Caiso_M, Dataset_Production_M, Dataset_Caiso, Dataset_Production, Dataset_Synthetic
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Custom_S, Dataset_wind, Dataset_Monash
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom, # Dataset_Custom_S,
    'wind': Dataset_wind,
    'monash': Dataset_Monash, 
    'caiso': Dataset_Caiso,
    'production': Dataset_Production, 
    'synthetic': Dataset_Synthetic, 
    'caiso_m': Dataset_Caiso_M, 
    'production_m': Dataset_Production_M, 
}

def data_provider(args, flag, return_full_data=False, shuffle_flag_train=True):

    if args.dataset_name in ['ECL','ETTh1','ETTh2','ETTm1','ETTm2','Exchange','traffic','weather','illnes','wind']:

        Data = data_dict[args.data]
        timeenc = 0 if args.embed != 'timeF' else 1

        if flag in ['test', 'val']:
            shuffle_flag = False
            drop_last = True
            batch_size = args.test_batch_size
            freq = args.freq
        else:
            shuffle_flag = shuffle_flag_train
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq

        data_set = Data(
            args, 
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq
        )

        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        args.dim_datetime_feats = data_set.dim_datetime_feats

        if return_full_data:
            return data_set, data_set.data_x
        else:
            return data_set, data_loader

    elif args.dataset_name in ['covid_deaths_dataset', 'sunspot_dataset_without_missing_values','elecdemand_dataset','saugeenday_dataset','wind_4_seconds_dataset','dominick_dataset','weather_dataset']:

        Data = data_dict['monash']

        if flag in ['test', 'val']:
            shuffle_flag = False
            drop_last = True
            batch_size = args.test_batch_size
        else:
            shuffle_flag = shuffle_flag_train
            drop_last = True
            batch_size = args.batch_size
            
        data_set = Data(
            args, 
            root_path=args.root_path,
            flag=flag
        )

        # data_set.__getitem__(1000)
        print(flag, len(data_set), data_set.global_inx)

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        args.dim_datetime_feats = data_set.dim_datetime_feats

        if return_full_data:
            return data_set, data_set.data_x
        else:
            return data_set, data_loader

    elif args.dataset_name in ['caiso', 'production', 'caiso_m', 'production_m', 'synthetic', 'system_KS']:

        Data = data_dict[args.data]

        if flag in ['test', 'val']:
            shuffle_flag = False
            drop_last = True
            batch_size = args.test_batch_size
            freq = args.freq
        else:
            shuffle_flag = shuffle_flag_train
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq

    data_set = Data(
        args, 
        root_path=args.root_path,
        flag=flag
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    
    if return_full_data:
        return data_set, data_set.values
    else:
        return data_set, data_loader


