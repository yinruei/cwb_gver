import pandas as pd
import numpy as np
import os
import xarray as xr
import torch
import torch.nn.functional as F
import click
import time
from math import sqrt
from collections import defaultdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

path = os.getcwd()
file_path = f'{path}/NCEPGEFS'

def read_data_for_date_range_f00(end_date, variable, ens):
    file = f'{file_path}/NCEPGEFS.{end_date}.mem{ens}.grib'
    try:
        if variable['typeOfLevel'] == 'isobaricInhPa':
            ds = xr.open_dataset(file, engine='cfgrib',
                backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'level': variable['level'],
                'shortName': variable['shortName'], 'stepRange': '0'
            }}).sel(latitude=slice(80, 20), longitude=slice(0, 360))
            key = variable['shortName']

        elif variable['typeOfLevel'] == 'heightAboveGround':
            ds = xr.open_dataset(file, engine='cfgrib',
                backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': variable['level'],
                'shortName': variable['shortName'], 'stepRange': '0'
            }}).sel(latitude=slice(80, 20), longitude=slice(0, 360))
            key = list(ds.keys())[0]

        data = ds.variables[key].data

        if len(data) == 0:
            return None

        return {'date': end_date, 'data': data}
    except Exception as e:
        return None

def read_data_for_dtg(dtg, step_range, variable, ens):
    current_date_str = dtg.strftime('%Y%m%d%H')
    file = f'{file_path}/NCEPGEFS.{current_date_str}.mem{ens}.grib'
    try:
        if variable['typeOfLevel'] == 'isobaricInhPa':
            ds = xr.open_dataset(file, engine='cfgrib',
                backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'level': variable['level'],
                    'shortName': variable['shortName'], 'stepRange': str(step_range),
                    }}).sel(latitude=slice(80, 20), longitude=slice(0, 360))
            key = variable['shortName']
        elif variable['typeOfLevel'] == 'heightAboveGround':
            ds = xr.open_dataset(file, engine='cfgrib',
                backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': variable['level'],
                    'shortName': variable['shortName'], 'stepRange': str(step_range),
                    }}).sel(latitude=slice(80, 20), longitude=slice(0, 360))
            key = list(ds.keys())[0]

        data = ds.variables[key].data
        if len(data) == 0:
            return None

        return {'date': current_date_str, 'stepRange': step_range, 'ens': ens, 'data': data}
    except FileNotFoundError:
        return None

def compute_metrics(data1, data2):
    average_data1 = data1['average_data']
    fcst = data1['stepRange']
    data2_data = data2['data']
    
    if average_data1.shape != data2_data.shape:
        raise ValueError("Data shapes do not match")
    
    average_data1_tensor = torch.tensor(average_data1, dtype=torch.float32).cuda()
    data2_data_tensor = torch.tensor(data2_data, dtype=torch.float32).cuda()

    rmse = sqrt(F.mse_loss(average_data1_tensor, data2_data_tensor).item())
    bias_tensor = torch.mean(average_data1_tensor - data2_data_tensor)
    mae_tensor = torch.mean(torch.abs(average_data1_tensor - data2_data_tensor))

    return rmse, bias_tensor, mae_tensor, fcst

def process_variable(variable):
    dtg_str = '2023101000'
    dtg = datetime.strptime(dtg_str, '%Y%m%d%H')
    new_dtg = dtg - timedelta(days=1)
    new_dtg_str = new_dtg.strftime('%Y%m%d%H')
    dtg = pd.to_datetime(new_dtg_str, format='%Y%m%d%H')
    initial_step_range = 24

    result = []
    with ThreadPoolExecutor() as executor:
        f00_future = executor.submit(read_data_for_date_range_f00, dtg_str, variable, 1)
        dtg_future = executor.submit(read_data_for_dtg, dtg, initial_step_range, variable, 1)

        f00_data = f00_future.result()
        dtg_data = dtg_future.result()

        if f00_data is not None and dtg_data is not None:
            data_array_tensor = torch.tensor([f00_data['data'], dtg_data['data']], dtype=torch.float32).cuda()
            average_data = torch.mean(data_array_tensor, dim=0).cpu().numpy()

            result.append({
                'date': dtg_data['date'],
                'average_data': average_data,
                'stepRange': dtg_data['stepRange']
            })

    data_f00 = read_data_for_date_range_f00(dtg_str, variable, 2)

    metrics_values = []
    for data1_dict in result:
        metrics = compute_metrics(data1_dict, data_f00[0])
        metrics_values.append({
            'rmse': metrics[0],
            'bias': metrics[1].cpu().numpy(),
            'mae': metrics[2].cpu().numpy(),
            'fcst': metrics[3],
            'level': variable['level'],
            'var': variable['shortName']
        })

    df_metrics = pd.DataFrame(metrics_values)

    rmse_output_file = f'rmse_{variable["shortName"]}_P{variable["level"]}_{dtg_str}.txt'
    bias_output_file = f'bias_{variable["shortName"]}_P{variable["level"]}_{dtg_str}.txt'
    mae_output_file = f'mae_{variable["shortName"]}_P{variable["level"]}_{dtg_str}.txt'

    df_metrics[['rmse', 'fcst', 'level', 'var']].to_csv(rmse_output_file, sep=' ', index=False)
    df_metrics[['bias', 'fcst', 'level', 'var']].to_csv(bias_output_file, sep=' ', index=False)
    df_metrics[['mae', 'fcst', 'level', 'var']].to_csv(mae_output_file, sep=' ', index=False)

start_time = time.time()

variables = [
    {'typeOfLevel': 'isobaricInhPa', 'level': 250, 'shortName': 'u'},
    {'typeOfLevel': 'isobaricInhPa', 'level': 250, 'shortName': 'v'},
    {'typeOfLevel': 'isobaricInhPa', 'level': 850, 'shortName': 'u'},
    {'typeOfLevel': 'isobaricInhPa', 'level': 850, 'shortName': 'v'},
    {'typeOfLevel': 'isobaricInhPa', 'level': 850, 'shortName': 't'},
    {'typeOfLevel': 'isobaricInhPa', 'level': 500, 'shortName': 'gh'},
    {'typeOfLevel': 'isobaricInhPa', 'level': 1000, 'shortName': 'gh'},
    {'typeOfLevel': 'heightAboveGround', 'level': 2, 'shortName': '2t'},
    {'typeOfLevel': 'heightAboveGround', 'level': 10, 'shortName': '10u'},
    {'typeOfLevel': 'heightAboveGround', 'level': 10, 'shortName': '10v'},
]

with ThreadPoolExecutor() as executor:
    executor.map(process_variable, variables)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
