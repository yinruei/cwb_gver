import pandas as pd
import numpy as np
import os
import time
import nvtx

# 記錄起始時間
start_time = time.time()


# Load the NumPy arrays
anl = np.load("2023090500_anl.npy")
fcst = np.load("2023090500.npy")

print('anl-->', anl.shape)
print('fcst-->', fcst.shape)

import jax
import jax.numpy as jnp

@jax.jit
def compute_metrics(aa_arr, ff_arr):
    # MSE
    mse_score = jnp.mean((aa_arr - ff_arr)**2)
    # RMSE
    rmse_score = jnp.sqrt(mse_score)
    # MAE
    mae_score = jnp.mean(jnp.abs(aa_arr - ff_arr))
    # Bias
    bias_score = jnp.mean(aa_arr - ff_arr)
    return mse_score, rmse_score, mae_score, bias_score

@jax.jit
def vectorize_compute_metrics(aa_arr, ff_arr):
    vmapped = jax.vmap(compute_metrics, in_axes=(None, 1))
    mse_score, rmse_score, mae_score, bias_score = \
        vmapped(aa_arr, ff_arr)
    return mse_score, rmse_score, mae_score, bias_score

anl_jnp_arr = jnp.asarray(anl)
fcst_tensor = jnp.asarray(fcst)

rmse = []
mae = []
bias = []
one_var_start_time = time.time()
for i in range(10):
    nvtx.push_range("Compute score", domain="vmap_compute_score")
    _, rmse_score, mae_score, bias_score = \
        vectorize_compute_metrics(anl_jnp_arr[:, i, :, :], fcst_tensor[:, i, :, :, :])
    rmse.append(rmse_score)
    mae.append(mae_score)
    bias.append(bias_score)

    nvtx.pop_range(domain="vmap_compute_score")
one_var_end_time = time.time()
print(f"One Variable Compute Teime: {one_var_end_time - one_var_start_time} sec")

# nvtx.push_range("Write to CSV", domain="write_to_csv")
# # 創建一個字典來保存指標數值
# metrics_values = {
#     'RMSE': rmse,
#     'MAE': mae,
#     'Bias': bias,
# }

# # 將字典轉換為 DataFrame
# df_metrics = pd.DataFrame(metrics_values)

# # 定義目標資料夾
# output_folder = "score"

# # 如果目標資料夾不存在，則創建它
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# # 將 DataFrame 寫入 txt 檔案
# txt_filename = os.path.join(output_folder, "metrics_results.txt")
# df_metrics.to_csv(txt_filename, sep=' ', index=False)

# print(f'Metrics results saved to {txt_filename}')

@jax.jit
def compute_predict_loss(pred, target, pred_obs, target_obs):
    losses = (target - pred) ** 2
    brier_score = jnp.mean(losses.mean())

    rpss_score = jnp.mean(1 - jnp.sum((pred_obs - target_obs) ** 2, axis=1) / jnp.sum((pred_obs ** 2), axis=1))
    crpss_score = jnp.mean((1 - jnp.sum((pred_obs - target_obs) ** 2, axis=1) / np.sum((pred_obs ** 2), axis=1)))
    return brier_score, rpss_score, crpss_score

# 假設你已經有實際觀測值和預測的機率，這裡使用隨機數生成
for i in range(5): # Assume 10 set of data
    key = jax.random.PRNGKey(0)
    subkeys = jax.random.split(key, 2)
    y = jax.random.randint(subkeys[0], (30, 10, 71, 73, 144), 0, 2 )
    y_preds = jax.random.normal(subkeys[1], (30, 10, 71, 73, 144), jnp.float32)

    key = jax.random.PRNGKey(1)
    subkeys = jax.random.split(key, 2)
    y_obs = jax.random.normal(subkeys[1], (30, 10, 71, 73, 144), jnp.float32)
    y_obs_preds = jax.random.normal(subkeys[1], (30, 10, 71, 73, 144), jnp.float32)

    pred_loss_start_time = time.time()
    brier_score, rpss_score, crpss_score = compute_predict_loss(y_preds, y, y_obs_preds, y_obs)
    pred_loss_end_time = time.time()

    print(f"Data {i}")
    print("Brier Score:", brier_score)
    print("RPSS:", rpss_score)
    print("CRPSS:", crpss_score)
    print("Compute Pred Loss Time:", pred_loss_end_time - pred_loss_start_time, " sec.")

# 記錄結束時間
end_time = time.time()

# # 計算總共花費的時間
execution_time = end_time - start_time
print(f"End2End Execution Time:{execution_time} Sec")
