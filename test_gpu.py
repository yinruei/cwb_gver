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


# Move NumPy arrays to PyTorch tensors on GPU
anl_tensor = torch.tensor(anl[:,:,:,:], dtype=torch.float32).cuda()
fcst_tensor = torch.tensor(fcst[:,0,:,:,:], dtype=torch.float32).cuda()


# # Calculate the squared differences
squared_diff = (anl_tensor - fcst_tensor) ** 2

# # Calculate the mean along the specified dimensions
# mse = torch.mean(squared_diff, dim=(0, 1, 2))

# # Calculate the square root to obtain RMSE
# rmse = torch.sqrt(mse)

# # Move the resulting RMSE tensor back to CPU if you want to use it in CPU operations
# rmse_cpu = rmse.cpu()

# # Print the resulting RMSE tensor
# print("RMSE:", rmse_cpu.item())