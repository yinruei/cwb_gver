import copy
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
import properscoring

# 記錄起始時間
start_time = time.time()

# variables = [
#     {'typeOfLevel': 'isobaricInhPa', 'level': 250, 'shortName': 'u'},
#     {'typeOfLevel': 'isobaricInhPa', 'level': 250, 'shortName': 'v'},
#     {'typeOfLevel': 'isobaricInhPa', 'level': 850, 'shortName': 'u'},
#     {'typeOfLevel': 'isobaricInhPa', 'level': 850, 'shortName': 'v'},
#     {'typeOfLevel': 'isobaricInhPa', 'level': 850, 'shortName': 't'},
#     {'typeOfLevel': 'isobaricInhPa', 'level': 500, 'shortName': 'gh'},
#     {'typeOfLevel': 'isobaricInhPa', 'level': 1000, 'shortName': 'gh'},
#     {'typeOfLevel': 'heightAboveGround', 'level': 2, 'shortName': '2t'},
#     {'typeOfLevel': 'heightAboveGround', 'level': 10, 'shortName': '10u'},
#     {'typeOfLevel': 'heightAboveGround', 'level': 10, 'shortName': '10v'},
# ]
# all_ = []
# for i in range(30):
#     file = f'NCEPGEFS/NCEPGEFS.2023090500.mem{i+1}.grib'
#     s_all_ = []
#     for variable in variables:
#         if os.path.exists(file) and os.path.getsize(file) > 0:
#             if variable['typeOfLevel'] == 'isobaricInhPa':
#                 ds = xr.open_dataset(file, engine='cfgrib',
#                     backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'level': variable['level'],
#                     'shortName': variable['shortName']
#                 }})
#                 key = variable['shortName']

#             elif variable['typeOfLevel'] == 'heightAboveGround':
#                 ds = xr.open_dataset(file, engine='cfgrib',
#                     backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': variable['level'],
#                     'shortName': variable['shortName']
#                 }})
#                 key = list(ds.keys())[0]

#             data = ds.variables[key].data
#             s_all_.append(data)
#     all_.append(copy.deepcopy(s_all_))
# all_ = np.array(all_)

# print('shape', all_.shape)

# np.save("2023090500.npy", all_)

# Move # Load the NumPy arrays
anl = np.load("2023090500_anl.npy")
fcst = np.load("2023090500.npy")

print('anl-->', anl.shape)
print('fcst-->', fcst.shape)

# 將NumPy數據轉換為PyTorch張量
anl_tensor = torch.from_numpy(anl)
fcst_tensor = torch.from_numpy(fcst)

# 將張量移動到GPU上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print('device--->', device)
rmse = []
mae = []
bias = []
for i in range(10):
    nvtx.push_range("One Variable", domain="variable")
    for j in range(0, 71, 2):
        nvtx.push_range("Compute score", domain="compute_score")
        aa = anl_tensor[:, i, :, :].to(device)
        ff = fcst_tensor[:, i, j, :, :].to(device)
        mse = F.mse_loss(ff, aa)
        rmse_value = torch.sqrt(mse)
        # print('i-->', rmse_value.item())  # 如果需要將結果轉換為 Python 數值
        rmse.append(rmse_value.item())
         # 計算 MAE
        mae_value = F.l1_loss(ff, aa)
        mae.append(mae_value.item())
        # 計算 Bias
        bias_value = torch.mean(ff - aa)
        bias.append(bias_value.item())

        nvtx.pop_range(domain="compute_score")

nvtx.push_range("Write to CSV", domain="write_to_csv")
# 創建一個字典來保存指標數值
metrics_values = {
    'RMSE': rmse,
    'MAE': mae,
    'Bias': bias,
}

# 將字典轉換為 DataFrame
df_metrics = pd.DataFrame(metrics_values)

# 定義目標資料夾
output_folder = "score"

# 如果目標資料夾不存在，則創建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 將 DataFrame 寫入 txt 檔案
txt_filename = os.path.join(output_folder, "metrics_results.txt")
df_metrics.to_csv(txt_filename, sep=' ', index=False)

print(f'Metrics results saved to {txt_filename}')



# 假設你已經有實際觀測值和預測的機率，這裡使用隨機數生成
y = torch.randint(0, 2, (30, 10, 71, 73, 144), dtype=torch.float32).cuda()  #  # 在 GPU 上生成實際觀測值
y_preds = torch.rand(30, 10, 71, 73, 144).cuda()  #  # 在 GPU 上生成預測的機率

# 計算 Brier Score
losses = (y - y_preds) ** 2
brier_score = losses.mean().item()

print("Brier Score:", brier_score)

# 假設你有觀測和預測的分布，這裡使用隨機數生成
y_obs = torch.rand(30, 10, 71, 73, 144).cuda()  #  # 在 GPU 上生成實際觀測的分布
y_probs = torch.rand(30, 10, 71, 73, 144).cuda()  #  # 在 GPU 上生成預測的分布

# 計算 RPSS
rpss = 1 - ((y_probs - y_obs) ** 2).sum(dim=1) / ((y_probs ** 2).sum(dim=1))

print("RPSS:", rpss.mean().item())

# 計算 CRPSS
crpss = (1 - ((y_probs - y_obs) ** 2).sum(dim=1) / ((y_probs ** 2).sum(dim=1))).mean()

print("CRPSS:", crpss.item())

# 記錄結束時間
end_time = time.time()

# 計算總共花費的時間
execution_time = end_time - start_time
print(f"程式執行時間：{execution_time} 秒")
