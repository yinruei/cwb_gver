import pandas as pd
import numpy as np
import os
import xarray as xr
import torch
import torch.nn.functional as F  # 使用PyTorch的functional模組
import click
import time
from sklearn.metrics import mean_squared_error
from math import sqrt
from collections import defaultdict
from datetime import datetime, timedelta


path = os.getcwd()
file_path = f'{path}/NCEPGEFS'

def read_data_for_date_range_f00(end_date, variable):
    # Convert the end_date to a pandas datetime object
    # end_date = pd.to_datetime(end_date, format='%Y%m%d%H')
    data_list = []
    for ens in range(1, 31):
        # Construct the file path for the current date
        file = f'{file_path}/NCEPGEFS.{end_date}.mem{ens}.grib'
        # Check if the file exists and has a non-zero size
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

                # Check if data is empty (contains no data)
                if len(data) == 0:
                    print(f'Date {end_date} has no data. Skipping...')
                    continue
                data_list.append({'date': end_date, 'data': data})
            except Exception as e:
                print(f'Error processing date {end_date}: {e}')
        else:
            print(f'File not found or empty for date {end_date}. Skipping...')

    # 创建一个字典来存储每个日期的数据
    date_data_dict = {}

    for entry in data_list:
        date = entry['date']
        data = entry['data']

        if date in date_data_dict:
            date_data_dict[date].append(data)
        else:
            date_data_dict[date] = [data]

    # 计算每个日期的数据平均值并存储在新的字典中
    average_data_dict = {}
    for date, data_list in date_data_dict.items():
        data_array = np.array(data_list)
        average_data = np.mean(data_array, axis=0)
        average_data_dict[date] = average_data

    # 计算每个日期的数据平均值并存储在新的字典中
    # average_data_dict = {}
    # for date, data_list in date_data_dict.items():
    #     # 將數據轉換為 PyTorch 張量並將其移至 GPU
    #     data_array_tensor = torch.tensor(data_list, dtype=torch.float32).cuda()
    #     average_data = torch.mean(data_array_tensor, dim=0).cpu().numpy()
    #     average_data_dict[date] = average_data
   

    # 打印或使用average_data_dict，它包含了每个日期的平均数据
    result = []
    for date, average_data in average_data_dict.items():
        # print(f"Date: {date}")
        # print("Average Data:", average_data)
        result.append({
            'date': date,
            'data': average_data
        })
    
    return result


def read_data_for_dtg(dtg, step_range, variable):
    data_list = []
    while step_range <= 840:
        current_date_str = dtg.strftime('%Y%m%d%H')
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
                        print(f'Date {current_date_str}, stepRange {step_range} for ens {ens} has no data. Skipping...')
                        continue
                    
                    # print('Date:', current_date_str, 'stepRange:', step_range, 'ens:', ens)
                    data_list.append({'date': current_date_str, 'stepRange': step_range, 'ens': ens, 'data': data})
                except FileNotFoundError:
                    print(f'File not found for date {current_date_str}, stepRange {step_range}, ens {ens}. Skipping...')
            else:
                print(f'File not found or empty for date {current_date_str}, stepRange {step_range}, ens {ens}. Skipping...')
        # print('data', data)
        
        # Decrement the dtg by one day (24 hours)
        dtg -= pd.Timedelta(hours=24)
        # Increment the step_range by 24 hours
        step_range += 24
    
    # 创建一个字典，用于按日期和时间步范围分组数据
    data_dict = defaultdict(list)

    # 将数据按日期和时间步范围分组
    for data_item in data_list:
        key = (data_item['date'], data_item['stepRange'])
        data_dict[key].append(data_item['data'])

    # 计算每组数据的平均值
    average_data_list = []
    for key, data_items in data_dict.items():
        date, step_range = key
        average_data = np.mean(data_items, axis=0)
        average_data_list.append({'date': date, 'stepRange': step_range, 'average_data': average_data})

    # 計算每組數據的平均值
    # average_data_list = []
    # for key, data_items in data_dict.items():
    #     date, step_range = key
    #     # 將數據轉換為 PyTorch 張量並將其移至 GPU
    #     data_items_tensor = torch.tensor(data_items, dtype=torch.float32).cuda()
    #     average_data = torch.mean(data_items_tensor, dim=0).cpu().numpy()
    #     average_data_list.append({'date': date, 'stepRange': step_range, 'average_data': average_data})

    return average_data_list

def compute_rmse(data1, data2):
    average_data1 = data1['average_data']
    fcst = data1['stepRange']
    data2_data = data2['data']
    
    # 確保兩個數據具有相同的形狀
    if average_data1.shape != data2_data.shape:
        raise ValueError("兩個數據形狀不匹配")
    
    # 計算均方根誤差（RMSE）
    rmse = sqrt(mean_squared_error(average_data1, data2_data))
 
    # 將average_data1和data2_data轉換為PyTorch張量並將它們移至GPU
    # average_data1_tensor = torch.tensor(average_data1, dtype=torch.float32).cuda()
    # data2_data_tensor = torch.tensor(data2_data, dtype=torch.float32).cuda()

    # # 計算均方根誤差（RMSE）
    # rmse = sqrt(F.mse_loss(average_data1_tensor, data2_data_tensor).item())

    # 修改為在 CPU 上進行計算
    # average_data1_tensor = torch.tensor(average_data1, dtype=torch.float32)
    # data2_data_tensor = torch.tensor(data2_data, dtype=torch.float32)
    # # 計算均方根誤差（RMSE）
    # rmse = sqrt(F.mse_loss(average_data1_tensor, data2_data_tensor).item())

    return rmse, fcst

    # 將average_data和data轉換為PyTorch張量並將它們移至GPU
    # average_data1 = torch.Tensor(data1['average_data']).to('cuda')
    # data2_data = torch.Tensor(data2['data']).to('cuda')
    # fcst = data1['stepRange']

    # # 確保兩個張量具有相同的形狀
    # if average_data1.shape != data2_data.shape:
    #     raise ValueError("兩個數據形狀不匹配")
    
    # # 計算均方根誤差（RMSE）在GPU上
    # rmse = sqrt(mean_squared_error(average_data1.cpu().numpy(), data2_data.cpu().numpy()))
    # print(f"RMSE 值：{rmse}")

    # return rmse, fcst

def compute_bias(data1, data2):
    average_data1 = data1['average_data']
    fcst = data1['stepRange']
    data2_data = data2['data']
    
    # 確保兩個數據具有相同的形狀
    if average_data1.shape != data2_data.shape:
        raise ValueError("兩個數據形狀不匹配")
    
    # Calculate bias
    bias = np.mean(average_data1 - data2_data)

    return bias, fcst
    
    # average_data1 = data1['average_data']
    # fcst = data1['stepRange']
    # data2_data = data2['data']

    # # 確保兩個數據具有相同的形狀
    # if average_data1.shape != data2_data.shape:
    #     raise ValueError("兩個數據形狀不匹配")

    # # 將數據轉換為 PyTorch 張量並將其移至 GPU
    # average_data1_tensor = torch.tensor(average_data1, dtype=torch.float32).cuda()
    # data2_data_tensor = torch.tensor(data2_data, dtype=torch.float32).cuda()

    # # 將數據轉換為 PyTorch 張量並將其移至 CPU
    # # average_data1_tensor = torch.tensor(average_data1, dtype=torch.float32)
    # # data2_data_tensor = torch.tensor(data2_data, dtype=torch.float32)
   
    # # 計算偏差
    # bias_tensor = torch.mean(average_data1_tensor - data2_data_tensor)

    # # 將偏差轉換回 NumPy 陣列
    # bias = bias_tensor.cpu().numpy()

    # return bias, fcst

def compute_mae(data1, data2):
    average_data1 = data1['average_data']
    fcst = data1['stepRange']
    data2_data = data2['data']
    
    # 確保兩個數據具有相同的形狀
    if average_data1.shape != data2_data.shape:
        raise ValueError("兩個數據形狀不匹配")
    
    # Calculate mae
    mae = np.mean(np.abs(average_data1 - data2_data))
    
    return mae, fcst

    # average_data1 = data1['average_data']
    # fcst = data1['stepRange']
    # data2_data = data2['data']

    # # 確保兩個數據具有相同的形狀
    # if average_data1.shape != data2_data.shape:
    #     raise ValueError("兩個數據形狀不匹配")

    # # 將數據轉換為 PyTorch 張量並將其移至 GPU
    # average_data1_tensor = torch.tensor(average_data1, dtype=torch.float32).cuda()
    # data2_data_tensor = torch.tensor(data2_data, dtype=torch.float32).cuda()

    # # 將數據轉換為 PyTorch 張量並將其移至 CPU
    # # average_data1_tensor = torch.tensor(average_data1, dtype=torch.float32)
    # # data2_data_tensor = torch.tensor(data2_data, dtype=torch.float32)
   
    # # 計算平均絕對誤差（MAE）
    # mae_tensor = torch.mean(torch.abs(average_data1_tensor - data2_data_tensor))

    # # 將 MAE 轉換回 NumPy 陣列
    # mae = mae_tensor.cpu().numpy()

    # return mae, fcst

# Check if CUDA (GPU support) is available
# if torch.cuda.is_available():
#     print("GPU is available.")
# else:
#     print("GPU is not available. Switching to CPU.")

# exit()
# 記錄開始時間W
start_time = time.time()
# Example usage:
# Specify the initial dtg and initial step_range
# 输入的日期时间字符串
dtg_str = '2023101000'

# 将输入字符串解析为 datetime 对象
dtg = datetime.strptime(dtg_str, '%Y%m%d%H')

# 减去一天
new_dtg = dtg - timedelta(days=1)

# 将结果转换为字符串
new_dtg_str = new_dtg.strftime('%Y%m%d%H')

print('new_dtg_str', new_dtg_str)

dtg = pd.to_datetime(new_dtg_str, format='%Y%m%d%H')
print('dtg', dtg)

initial_step_range = 24

# 定義要處理的不同 level 和 shortName 的變數
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

    result = read_data_for_dtg(dtg, initial_step_range, variable)

    data_f00 = read_data_for_date_range_f00(dtg_str, variable)

    rmse_values = []
    bias_values = []
    mae_values = []
    for data1_dict in result:
        rmse, fcst = compute_rmse(data1_dict, data_f00[0])  # 假設data2只有一個字典
        rmse_values.append({
            'rmse': rmse,
            'fcst': fcst,
            'level': variable['level'],
            'var': variable['shortName']
        })

        bias, fcst = compute_bias(data1_dict, data_f00[0])  # Assuming data_f00 has only one dictionary
        bias_values.append({
            'bias': bias,
            'fcst': fcst,
            'level': variable['level'],
            'var': variable['shortName']
        })

        mae, fcst = compute_mae(data1_dict, data_f00[0])  # Assuming data_f00 has only one dictionary
        mae_values.append({
            'mae': mae,
            'fcst': fcst,
            'level': variable['level'],
            'var': variable['shortName']
        })



    # 創建Pandas DataFrame
    df_rmse = pd.DataFrame(rmse_values)
    df_bias = pd.DataFrame(bias_values)
    df_mae = pd.DataFrame(mae_values)

    # 将DataFrame保存为以空白分隔的文本文件
    rmse_output_file = f'rmse_{variable["shortName"]}_P{variable["level"]}_{dtg_str}.txt'
    bias_output_file = f'bias_{variable["shortName"]}_P{variable["level"]}_{dtg_str}.txt'
    mae_output_file = f'mae_{variable["shortName"]}_P{variable["level"]}_{dtg_str}.txt'
    df_rmse.to_csv(rmse_output_file, sep=' ', index=False)
    df_bias.to_csv(bias_output_file, sep=' ', index=False)
    df_mae.to_csv(mae_output_file, sep=' ', index=False)

# 記錄結束時間
end_time = time.time()

# 計算總共花費的時間
execution_time = end_time - start_time
print(f"程式執行時間：{execution_time} 秒")

