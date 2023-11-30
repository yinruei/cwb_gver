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
import nvtx

path = os.getcwd()
file_path = f'{path}/NCEPGEFS'

def read_data_for_date_range_f00(end_date, variable):
    data_list = []
    # for ens in range(1, 31):
        # file = f'{file_path}/NCEPGEFS.{end_date}.mem{ens}.grib'
    file = f'{file_path}/NCEPGEFS.{end_date}.mem1.grib'
    if os.path.exists(file) and os.path.getsize(file) > 0:
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

            # print('data--->', data)
            # if len(data) == 0:
            #     continue
            data_list.append({'date': end_date, 'data': data})
        except Exception as e:
            pass
    else:
        pass

    date_data_dict = {}

    for entry in data_list:
        date = entry['date']
        data = entry['data']

        if date in date_data_dict:
            date_data_dict[date].append(data)
        else:
            date_data_dict[date] = [data]

    average_data_dict = {}
    for date, data_list in date_data_dict.items():
        data_array_tensor = torch.tensor(data_list, dtype=torch.float32).cuda()
        average_data = torch.mean(data_array_tensor, dim=0).cpu().numpy()
        average_data_dict[date] = average_data

    result = []
    for date, average_data in average_data_dict.items():
        result.append({
            'date': date,
            'data': average_data
        })
    
    return result

def read_data_for_dtg(dtg, step_range, variable):
    data_list = []
    outer_start_time = time.time()
    while step_range <= 840:
        current_date_str = dtg.strftime('%Y%m%d%H')
        # print('step_range-->', step_range)
        # print('current_date_str-->', current_date_str)
        inner_start_time = time.time()
        for ens in range(1, 31):
            file = f'{file_path}/NCEPGEFS.{current_date_str}.mem{ens}.grib'
            if os.path.exists(file) and os.path.getsize(file) > 0:
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
                        continue
                    data_list.append({'date': current_date_str, 'stepRange': step_range, 'ens': ens, 'data': data})
                except FileNotFoundError:
                    pass
            else:
                pass
        # inner_end_time = time.time()
        # inner_execution_time = inner_end_time - inner_start_time
        # print(f"inner_execution_time : {inner_execution_time} seconds")
        dtg -= pd.Timedelta(hours=24)
        step_range += 24
    
    # outer_end_time = time.time()
    # outer_execution_time = outer_end_time - outer_start_time
    # print(f"outer_execution_time : {outer_execution_time} seconds")
    
    data_dict = defaultdict(list)

    for data_item in data_list:
        key = (data_item['date'], data_item['stepRange'])
        data_dict[key].append(data_item['data'])

    average_data_list = []
    for key, data_items in data_dict.items():
        date, step_range = key
        data_items_tensor = torch.tensor(data_items, dtype=torch.float32).cuda()
        average_data = torch.mean(data_items_tensor, dim=0).cpu().numpy()
        average_data_list.append({'date': date, 'stepRange': step_range, 'average_data': average_data})

    return average_data_list

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

start_time = time.time()

dtg_str = '2023101000'
dtg = datetime.strptime(dtg_str, '%Y%m%d%H')
new_dtg = dtg - timedelta(days=1)
new_dtg_str = new_dtg.strftime('%Y%m%d%H')
dtg = pd.to_datetime(new_dtg_str, format='%Y%m%d%H')
initial_step_range = 24

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

for variable in variables:
    nvtx.push_range("One Variable", domain="variable")
    nvtx.push_range("Read DTG", domain="read_dtg")

    result = read_data_for_dtg(dtg, initial_step_range, variable)
    nvtx.pop_range(domain="read_dtg")
    nvtx.push_range("Read f00", domain="read_f00")
    data_f00 = read_data_for_date_range_f00(dtg_str, variable)
    nvtx.pop_range(domain="read_f00")


    metrics_values = []
    for data1_dict in result:
        nvtx.push_range("Compute score", domain="compute_score")
        metrics = compute_metrics(data1_dict, data_f00[0])
        nvtx.pop_range(domain="compute_score")
        nvtx.push_range("compute_score Result Append", domain="compute_score_r_append")

        metrics_values.append({
            'rmse': metrics[0],
            'bias': metrics[1].cpu().numpy(),
            'mae': metrics[2].cpu().numpy(),
            'fcst': metrics[3],
            'level': variable['level'],
            'var': variable['shortName']
        })
        nvtx.pop_range(domain="compute_score_r_append")

    nvtx.push_range("Write to CSV", domain="write_to_csv")
    df_metrics = pd.DataFrame(metrics_values)

    rmse_output_file = f'rmse_{variable["shortName"]}_P{variable["level"]}_{dtg_str}.txt'
    bias_output_file = f'bias_{variable["shortName"]}_P{variable["level"]}_{dtg_str}.txt'
    mae_output_file = f'mae_{variable["shortName"]}_P{variable["level"]}_{dtg_str}.txt'

    df_metrics[['rmse', 'fcst', 'level', 'var']].to_csv(rmse_output_file, sep=' ', index=False)
    df_metrics[['bias', 'fcst', 'level', 'var']].to_csv(bias_output_file, sep=' ', index=False)
    df_metrics[['mae', 'fcst', 'level', 'var']].to_csv(mae_output_file, sep=' ', index=False)
    nvtx.pop_range(domain="write_to_csv")
    nvtx.pop_range(domain="variable")

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

