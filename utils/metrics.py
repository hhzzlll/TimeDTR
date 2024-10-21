import numpy as np
import torch


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean()   # .mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

import numpy as np

Forecast = np.ndarray
Target = np.ndarray


def mase(forecast: Forecast, insample: np.ndarray, outsample: Target, frequency: int) -> np.ndarray:
    """
    MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf
    :param forecast: Forecast values. Shape: batch, time_o
    :param insample: Insample values. Shape: batch, time_i
    :param outsample: Target values. Shape: batch, time_o
    :param frequency: Frequency value
    :return: Same shape array with error calculated for each time step
    """
    return np.mean(np.abs(forecast - outsample)) / np.mean(np.abs(insample[:-frequency] - insample[frequency:]))


def nd(forecast: Forecast, target: Target) -> float:
    """
    Normalized deviation as defined in https://www.cs.utexas.edu/~rofuyu/papers/tr-mf-nips.pdf
    :param forecast: Forecast values. Shape: batch, time
    :param target: Target values. Shape: batch, time
    :return: Error value
    """
    return np.mean(np.abs(target - forecast)) / np.mean(np.abs(target))


def nrmse(forecast: Forecast, target: Target) -> float:
    """
    Normalized RMSE as defined in https://www.cs.utexas.edu/~rofuyu/papers/tr-mf-nips.pdf
    :param forecast: Forecast values. Shape: batch, time
    :param target: Target values. Shape: batch, time
    :return: Error values
    """
    return np.sqrt(np.mean(np.power((forecast - target), 2))) / (np.mean(np.abs(target)))
     

def mape(forecast: Forecast, target: Target) -> np.ndarray:
    """
    MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    :param forecast: Predicted values.
    :param target: Target values.
    :return: Same shape array with error calculated for each time step
    """
    return np.abs(forecast - target) / target


def smape_1(forecast: Forecast, target: Target) -> np.ndarray:
    """
    sMAPE loss as defined in "Appendix A" of
    http://www.forecastingprinciples.com/files/pdf/Makridakia-The%20M3%20Competition.pdf
    :param forecast: Forecast values. Shape: batch, time
    :param target: Target values. Shape: batch, time
    :return: Same shape array with error calculated for each time step
    """
    return 200 * np.abs(forecast - target) / (target + forecast)


def smape_2(forecast: Forecast, target: Target) -> np.ndarray:
    """
    sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)
    :param forecast: Forecast values. Shape: batch, time
    :param target: Target values. Shape: batch, time
    :return: Same shape array with sMAPE calculated for each time step of each timeseries.
    """
    denom = np.abs(target) + np.abs(forecast)
    # divide by 1.0 instead of 0.0, in case when denom is zero the enumerator will be 0.0 anyway.
    denom[denom == 0.0] = 1.0
    return 200 * np.abs(forecast - target) / denom

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    corr = CORR(pred, true)

    return mae, mse, rmse, mape, mspe, corr


def quantile_loss(target, forecast, q: float) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * ((target <= forecast) * 1.0 - q))
    )

def calc_quantile_CRPS(all_preds, true): 

    target = torch.tensor(true).cuda()
    forecast = torch.tensor(all_preds).cuda()

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = torch.sum(torch.abs(target))
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i])
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)




