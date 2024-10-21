
import os
import argparse
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

parser = argparse.ArgumentParser(description='Non-stationary Diffusion for Time Series Forecasting')

# basic config
parser.add_argument('--ii', type=int, default=0)
parser.add_argument('--use_window_normalization', type=bool, default=True)

parser.add_argument('--stage_mode', type=str, default="TWO", help="ONE, TWO")
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--out_figures', type=int, default=1)
parser.add_argument('--vis_ar_part', type=int, default=0, help='status')
parser.add_argument('--vis_MTS_analysis', type=int, default=1, help='status')

parser.add_argument('--model', type=str, default='DDPM', 
    help='model name, options: [DDPM]')

parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
parser.add_argument('--pretrain_epochs', type=int, default=5, help='train epochs')

parser.add_argument('--sample_times', type=int, default=1)
parser.add_argument('--beta_dist_alpha', type=float, default=-1)  # -1
parser.add_argument('--our_ddpm_clip', type=float, default=100) # 100

# data loader
parser.add_argument('--seq_len', type=int, default=192, help='input sequence length')
parser.add_argument('--label_len', type=int, default=7, help='start token length')
parser.add_argument('--pred_len', type=int, default=14, help='prediction sequence length')


parser.add_argument('--dataset_name', type=str, default='Exchange')
parser.add_argument('--weather_type', type=str, default='mintemp', help="['rain' 'mintemp' 'maxtemp' 'solar']")

parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--num_vars', type=int, default=8, help='encoder input size')

parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

# Diffusion Models
parser.add_argument('--interval', type=int, default=1000, help='number of diffusion steps')
parser.add_argument('--ot-ode', default=True, help='use OT-ODE model')
parser.add_argument("--beta-max", type=float, default=0.3, help="max diffusion for the diffusion model")
parser.add_argument("--t0", type=float, default=1e-4, help="sigma start time in network parametrization")
parser.add_argument("--T", type=float, default=1., help="sigma end time in network parametrization")
parser.add_argument('--model_channels', type=int, default=256)#256
parser.add_argument('--nfe', type=int, default=100)
parser.add_argument('--dim_LSTM', type=int, default=64)

parser.add_argument('--diff_steps', type=int, default=100, help='number of diffusion steps')
parser.add_argument('--UNet_Type', type=str, default='CNN', help=['CNN'])
parser.add_argument('--D3PM_kernel_size', type=int, default=5)
parser.add_argument('--use_freq_enhance', type=int, default=0)
parser.add_argument('--type_sampler', type=str, default='none', help=["none", "dpm"])
parser.add_argument('--parameterization', type=str, default='x_start', help=["noise", "x_start"])

parser.add_argument('--ddpm_inp_embed', type=int, default=256)#256
parser.add_argument('--ddpm_dim_diff_steps', type=int, default=256)#256
parser.add_argument('--ddpm_channels_conv', type=int, default=256)#256
parser.add_argument('--ddpm_channels_fusion_I', type=int, default=256)#256
parser.add_argument('--ddpm_layers_inp', type=int, default=5)
parser.add_argument('--ddpm_layers_I', type=int, default=5)
parser.add_argument('--ddpm_layers_II', type=int, default=5)
parser.add_argument('--cond_ddpm_num_layers', type=int, default=5)
parser.add_argument('--cond_ddpm_channels_conv', type=int, default=64)

parser.add_argument('--ablation_study_case', type=str, default="none", help="none, mix_1, ar_1, mix_ar_0, w_pred_loss")
parser.add_argument('--weight_pred_loss', type=float, default=0.0)
parser.add_argument('--ablation_study_F_type', type=str, default="CNN", help="Linear, CNN")
parser.add_argument('--ablation_study_masking_type', type=str, default="none", help="none, hard, segment")
parser.add_argument('--ablation_study_masking_tau', type=float, default=0.9)

# forecasting task

parser.add_argument('--learning_rate', type=float, default=0.0006, help='optimizer learning rate')

parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--seed', type=int, default=2024, help='random seed')

parser.add_argument('--num_workers', type=int, default=8, help='data loader num workers')
parser.add_argument('--itr', type=int, default=10, help='experiments times')
parser.add_argument('--batch_size', type=int, default=64, help='32 batch size of train input data')  # 32
parser.add_argument('--test_batch_size', type=int, default=64, help='32 batch size of train input data')  # 32

# parser.add_argument('--lradj', type=str, default='type2', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

parser.add_argument('--tag', type=str, default='')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

fix_seed = args.seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

if args.use_gpu:
    if args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    else:
        torch.cuda.set_device(args.gpu)

args.DATAdir = "../datasets"
args.data = "custom"
if args.dataset_name == "ECL":
    args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
    args.root_path = os.path.join(args.DATAdir, 'prediction/data_autoformer/electricity/')
    args.data_path = 'electricity.csv'
    args.use_valset = True
if args.dataset_name == "ETTh1":
    args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
    args.root_path = os.path.join(args.DATAdir, 'prediction/data_autoformer/ETT-small/')
    args.data_path = 'ETTh1.csv'
    args.data = "ETTh1"
    args.use_valset = True
if args.dataset_name == "ETTh2":
    args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
    args.root_path = os.path.join(args.DATAdir, 'prediction/data_autoformer/ETT-small/')
    args.data_path = 'ETTh2.csv'
    args.data = "ETTh2"
    args.use_valset = True
if args.dataset_name == "ETTm1":
    args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
    args.root_path = os.path.join(args.DATAdir, 'prediction/data_autoformer/ETT-small/')
    args.data_path = 'ETTm1.csv'
    args.data = "ETTm1"
    args.use_valset = True
if args.dataset_name == "ETTm2":
    args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
    args.root_path = os.path.join(args.DATAdir, 'prediction/data_autoformer/ETT-small/')
    args.data_path = 'ETTm2.csv'
    args.data = "ETTm2"
    args.use_valset = True
if args.dataset_name == "Exchange":
    args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
    args.root_path = os.path.join(args.DATAdir, 'prediction/data_autoformer/exchange_rate/')
    args.data_path = 'exchange_rate.csv'
    args.use_valset = True
if args.dataset_name == "traffic":
    args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
    args.root_path = os.path.join(args.DATAdir, 'prediction/data_autoformer/traffic/')
    args.data_path = 'traffic.csv'
    args.use_valset = True
if args.dataset_name == "weather":
    args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
    args.root_path = os.path.join(args.DATAdir, 'prediction/data_autoformer/weather/')
    args.data_path = 'weather.csv'
    args.use_valset = True
if args.dataset_name == "wind":
    args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
    args.root_path = os.path.join(args.DATAdir, 'prediction/data_autoformer/wind/')
    args.data_path = 'wind.csv'
    args.use_valset = True
    args.data = "wind"
    args.target = 'target'
if args.dataset_name == "illness":
    args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
    args.root_path = os.path.join(args.DATAdir, 'prediction/data_autoformer/illness/')
    args.data_path = 'national_illness.csv'
    args.use_valset = True

if args.dataset_name in ["covid_deaths_dataset","sunspot_dataset_without_missing_values","elecdemand_dataset","saugeenday_dataset","wind_4_seconds_dataset","dominick_dataset","weather_dataset"]:
    args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
    args.root_path = os.path.join(args.DATAdir, 'prediction/data_monash')
    args.data_path = ''
    args.use_valset = True

if args.dataset_name in ["caiso", "caiso_m"]:
    args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
    args.root_path = os.path.join(args.DATAdir, 'prediction/data_depts/caiso/')
    args.data_path = ''
    args.data = args.dataset_name
    args.use_valset = True
if args.dataset_name in ["production", "production_m"]:
    args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
    args.root_path = os.path.join(args.DATAdir, 'prediction/data_depts/nordpool/')
    args.data_path = ''
    args.data = args.dataset_name
    args.use_valset = True
if args.dataset_name == "synthetic":
    args.synthetic_mode = 'L'  # ['L', 'Q', 'C', 'LT', 'QT', 'CT']
    args.model_id = "{}_{}_{}_{}".format(args.dataset_name, args.synthetic_mode, args.seq_len, args.pred_len)
    args.root_path = os.path.join(args.DATAdir, 'prediction/data_depts/synthetic/')
    args.data_path = ''
    args.data = 'synthetic'
    args.use_valset = True
if args.dataset_name == "system_KS":
    args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
    args.root_path = os.path.join(args.DATAdir, 'prediction/data_depts/dynamic_KS/')
    args.data_path = ''
    args.data = 'system_KS'
    args.use_valset = True

print('Args in experiment:')
print(args)

Exp = Exp_Main

mae_, mse_, rmse_, mape_, mspe_, corr_ = [], [], [], [], [], []
if args.is_training:

    for ii in range(args.itr):

        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dt{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len, 
            ii,
            args.stage_mode)

        if args.tag != '':
            setting += '_' + str(args.tag)

        if args.ablation_study_case != "none":
            setting += '_' + str(args.tag)

        exp = Exp(args)

        # ==============================
        # Pertraining 
        # ==============================
        if args.stage_mode == 'TWO':
            print('>>>>>>>start pretraining : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            if args.model in ["MATS", "MATS2"]:
                # exp.mats_pretrain(setting)
                pass
            else:
                exp.pretrain(setting)
                pass

        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        if args.model == "D3VAE":
            exp.D3VAE_train(setting)
        else:
            exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        mae, mse, rmse, mape, mspe, corr = exp.test(setting, mode="test")
        mae_.append(mae)
        mse_.append(mse)
        rmse_.append(rmse)
        mape_.append(mape)
        mspe_.append(mspe)
        corr_.append(corr)

        torch.cuda.empty_cache()

        print('Final mean normed: ')
        print('> mae:{:.4f}, std:{:.4f}'.format(np.mean(mae_), np.std(mae_)))
        print('> mse:{:.4f}, std:{:.4f}'.format(np.mean(mse_), np.std(mse_)))
        print('> rmse:{:.4f}, std:{:.4f}'.format(np.mean(rmse_), np.std(rmse_)))
        print('> mape:{:.4f}, std:{:.4f}'.format(np.mean(mape_), np.std(mape_)))
        print('> mspe:{:.4f}, std:{:.4f}'.format(np.mean(mspe_), np.std(mspe_)))
        print('> corr:{:.4f}, std:{:.4f}'.format(np.mean(corr_), np.std(corr_)))


else:
    ii = args.ii

    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dt{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            ii,
            args.stage_mode)

    if args.tag != '':
        setting += '_' + str(args.tag)

    exp = Exp(args)
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    mae, mse, rmse, mape, mspe, corr = exp.test(setting, mode="test")
    mae_.append(mae)
    mse_.append(mse)
    rmse_.append(rmse)
    mape_.append(mape)
    mspe_.append(mspe)
    corr_.append(corr)

    torch.cuda.empty_cache()

    print('Final mean normed: ')
    print('> mae:{:.4f}, std:{:.4f}'.format(np.mean(mae_), np.std(mae_)))
    print('> mse:{:.4f}, std:{:.4f}'.format(np.mean(mse_), np.std(mse_)))
    print('> rmse:{:.4f}, std:{:.4f}'.format(np.mean(rmse_), np.std(rmse_)))
    print('> mape:{:.4f}, std:{:.4f}'.format(np.mean(mape_), np.std(mape_)))
    print('> mspe:{:.4f}, std:{:.4f}'.format(np.mean(mspe_), np.std(mspe_)))
    print('> corr:{:.4f}, std:{:.4f}'.format(np.mean(corr_), np.std(corr_)))

