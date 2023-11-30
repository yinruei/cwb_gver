import pandas as pd
import numpy as np
import os
import xarray as xr
import torch
import torch.nn.functional as F
import click
import cf2cdm
import time
from math import sqrt
from collections import defaultdict
from datetime import datetime, timedelta

path = os.getcwd()
file_path = f'{path}/NCEPGEFS'

# Set display options to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

variables_to_extract  = [
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


def read_data_for_date_range_f00(end_date):
    data_list = []
    # for ens in range(1, 31):
        # file = f'{file_path}/NCEPGEFS.{end_date}.mem{ens}.grib'
    # file = f'{file_path}/NCEPGEFS.{end_date}.mem1.grib'
    file = f'{file_path}/NCEPGEFS.{end_date}.grib'
    if os.path.exists(file) and os.path.getsize(file) > 0:
        try:
            
            ds = xr.open_dataset(file, engine='cfgrib')
            print('ds', len(ds),dir(ds), ds.dims)
            # Extract data for the first level of isobaricInhPa
            isobaricInhPa_level1_data = ds.sel(isobaricInhPa=ds.isobaricInhPa[0])
            print('isobaricInhPa_level1_data--->', isobaricInhPa_level1_data)
            # # Extract data for the second level of isobaricInhPa
            # isobaricInhPa_level2_data = ds.sel(isobaricInhPa=ds.isobaricInhPa[1])
            # print('isobaricInhPa_level2_data--->', isobaricInhPa_level2_data)

            # isobaricInhPa_level3_data = ds.sel(step='0 days')['t2m'].to_dataframe()
            # print('isobaricInhPa_level3_data--->', isobaricInhPa_level3_data)

            
            # print('ds---?', ds, type(ds), dir(ds))
            # data = cf2cdm.translate_coords(ds)

            # data = ds.variables
            # print('ds--->', ds)
            # print('data1--->', data.keys())
            # print('data2--->', data)

            # for k in data.keys():
            #     print('k-->', k)


            # # Select data for the specified isobaricInhPa and step
            # u850_subset_data = ds.sel(isobaricInhPa=850, step='0 days')['u'].to_dataframe()
            # v850_subset_data = ds.sel(isobaricInhPa=850, step='0 days')['v'].to_dataframe()
            # u250_subset_data = ds.sel(isobaricInhPa=250, step='0 days')['u'].to_dataframe()
            # v250_subset_data = ds.sel(isobaricInhPa=250, step='0 days')['v'].to_dataframe()
            # print('v250_subset_data-->', v250_subset_data)
            # # t850_subset_data = data.sel(isobaricInhPa=850, step='0 days')['t'].to_dataframe()
            # # gh500_subset_data = data.sel(isobaricInhPa=500, step='0 days')['gh'].to_dataframe()
            # # gh1000_subset_data = data.sel(isobaricInhPa=1000, step='0 days')['gh'].to_dataframe()
            # # t2m_subset_data = data.sel(step='0 days')['t2m'].to_dataframe()
            # # u10_subset_data = data.sel(step='0 days')['u10'].to_dataframe()
            # # v10_subset_data = data.sel(step='0 days')['v10'].to_dataframe()
            # # print('u850_subset_data', u850_subset_data)

            # # Select data for the specified isobaricInhPa and step
            # subset_data = data.sel(isobaricInhPa=850, step='0 days')
            # # subset_data2= data.sel(isobaricInhPa=500, step='0 days')
            # # gh_data = subset_data2['gh']
            # # Access the specific data variables
            # u_data = subset_data['u']
            # v_data = subset_data['v']
            # print('u_data--->', u_data, type(u_data))
            # print('v_data--->', v_data, type(v_data))
            # # print('gh_data--->', gh_data, type(gh_data))


            # # Calculate the mean along specified dimensions (e.g., latitude and longitude)
            # mean_u = u_data.mean()
            # mean_v = v_data.mean()
            # # mean_gh = gh_data.mean()

            # # Print or further process the mean DataArrays
            # print('Mean u_data along latitude and longitude:')
            # print(mean_u)

            # print('\nMean v_data along latitude and longitude:')
            # print(mean_v)

            # print('\nMean gh_data along latitude and longitude:')
            # print(mean_gh)




            # print('data--->', data)
            # if len(data) == 0:
            #     continue
            # data_list.append({'date': end_date, 'data': data})
        except Exception as e:
            pass
    else:
        pass


def read_data_for_dtg(dtg, step_range):
    data_list = []
    outer_start_time = time.time()
    while step_range <= 840:
        current_date_str = dtg.strftime('%Y%m%d%H')
        # print('step_range-->', step_range)
        # print('current_date_str-->', current_date_str)
        inner_start_time = time.time()
        # for ens in range(1, 31):
        file = f'{file_path}/NCEPGEFS.{current_date_str}.grib'
        if os.path.exists(file) and os.path.getsize(file) > 0:
            try:
                ds = xr.open_dataset(file, engine='cfgrib')
                data = cf2cdm.translate_coords(ds)

                # data = ds.variables
                print('ds--->', ds)
                print('data--->', data)
                # data = ds.variables[key].data
                # if len(data) == 0:
                #     continue
                # data_list.append({'date': current_date_str, 'stepRange': step_range, 'ens': ens, 'data': data})
            except FileNotFoundError:
                pass
        else:
            pass
        inner_end_time = time.time()
        inner_execution_time = inner_end_time - inner_start_time
        print(f"inner_execution_time : {inner_execution_time} seconds")
        dtg -= pd.Timedelta(hours=24)
        step_range += 24
    
    outer_end_time = time.time()
    outer_execution_time = outer_end_time - outer_start_time
    print(f"outer_execution_time : {outer_execution_time} seconds")
    

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

# result = read_data_for_dtg(dtg, initial_step_range)
data_f00 = read_data_for_date_range_f00(dtg_str)

# metrics_values = []
# for data1_dict in result:
#     metrics = compute_metrics(data1_dict, data_f00[0])
#     metrics_values.append({
#         'rmse': metrics[0],
#         'bias': metrics[1].cpu().numpy(),
#         'mae': metrics[2].cpu().numpy(),
#         'fcst': metrics[3],
#         'level': variable['level'],
#         'var': variable['shortName']
#     })

# df_metrics = pd.DataFrame(metrics_values)


end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

