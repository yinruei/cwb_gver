import os
import numpy as np
import time
import nvtx
import jax
import jax.numpy as jnp
from sklearn.metrics import roc_auc_score


IS_CPU = int(os.environ.get("IS_CPU", 0))
if IS_CPU >= 1:
    jax.config.update('jax_platform_name', 'cpu')


# Load the NumPy arrays
anl = np.load("2023090500_anl.npy")
fcst = np.load("2023090500.npy")

print('anl-->', anl.shape)
print('fcst-->', fcst.shape)

@jax.jit
def compute_metrics(ff_arr, aa_arr):
    # MSE
    mse_score = jnp.mean((ff_arr - aa_arr)**2)
    # RMSE
    rmse_score = jnp.sqrt(mse_score)
    # MAE
    mae_score = jnp.mean(jnp.abs(ff_arr - aa_arr))
    # Bias
    bias_score = jnp.mean(ff_arr - aa_arr)

    # Standard Deviation (as a measure of spread)
    spread_score = jnp.std(ff_arr - aa_arr)

    return mse_score, rmse_score, mae_score, bias_score, spread_score

@jax.jit
def vectorize_compute_metrics(ff_arr, aa_arr):
    vmapped = jax.vmap(compute_metrics, in_axes=(1, None))
    mse_score, rmse_score, mae_score, bias_score, spread_score = \
        vmapped(ff_arr, aa_arr)
    return mse_score, rmse_score, mae_score, bias_score, spread_score

anl_jnp_arr = jnp.asarray(anl)
fcst_tensor = jnp.asarray(fcst)

rmse = []
mae = []
bias = []
spread = []
one_var_start_time = time.time()
for i in range(10):
    if i == 1:
        one_var_start_after_compiling_time = time.time()
    nvtx.push_range("Compute score", domain="vmap_compute_score")
    _, rmse_score, mae_score, bias_score, spread_score = \
        vectorize_compute_metrics(fcst_tensor[:, i, :, :, :], anl_jnp_arr[:, i, :, :])
    rmse.append(rmse_score.block_until_ready())
    mae.append(mae_score.block_until_ready())
    bias.append(bias_score.block_until_ready())
    spread.append(spread_score.block_until_ready())
    nvtx.pop_range(domain="vmap_compute_score")
one_var_end_time = time.time()
one_var_time = one_var_end_time - one_var_start_time
one_var_after_compiling_time = one_var_end_time - one_var_start_after_compiling_time
print(f"One Variable Compute Time: {one_var_time} sec")
print(f"One Variable Compute (Ignore Compiling) Time: {one_var_after_compiling_time} sec")

@jax.jit
def compute_predict_loss(pred, target, pred_obs, target_obs):
    losses = (target - pred) ** 2
    brier_score = jnp.mean(losses.mean())

    rpss_score = jnp.mean(1 - jnp.sum((pred_obs - target_obs) ** 2, axis=1) / jnp.sum((pred_obs ** 2), axis=1))
    crpss_score = jnp.mean((1 - jnp.sum((pred_obs - target_obs) ** 2, axis=1) / np.sum((pred_obs ** 2), axis=1)))

    # Calculate CRPS (Continuous Ranked Probability Score)
    crps_score = jnp.mean(jnp.sum((pred_obs - target_obs) ** 2, axis=1))

    # Brier Skill Score
    bss_score = 1 - (crps_score / brier_score)

    # Histogram Distribution calculation
    histogram_bins = 10  # Choose an appropriate number of bins
    hist_pred_obs, _ = jnp.histogram(pred_obs, bins=histogram_bins)
    hist_target_obs, _ = jnp.histogram(target_obs, bins=histogram_bins)
    histogram_score = jnp.mean((hist_pred_obs - hist_target_obs) ** 2)

    return brier_score, bss_score, rpss_score, crpss_score, crps_score, histogram_score

# 假設你已經有實際觀測值和預測的機率，這裡使用隨機數生成
brier = []
rpss = []
crpss = []
bss = []
crps = []
# roc = []
histogram = []

total_pred_loss_time = 0
total_pred_loss_after_compiling_time = 0
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
    nvtx.push_range("Compute Pred Loss", domain="compute_pred_loss")
    brier_score, bss_score, rpss_score, crpss_score, crps_score, histogram_score = compute_predict_loss(y_preds, y, y_obs_preds, y_obs)

    brier.append(brier_score.block_until_ready())
    rpss.append(rpss_score.block_until_ready())
    crpss.append(crpss_score.block_until_ready())
    bss.append(bss_score.block_until_ready())
    crps.append(crps_score.block_until_ready())
    # roc.append(roc_area_score.block_until_ready())
    histogram.append(histogram_score.block_until_ready())

    nvtx.pop_range(domain="compute_pred_loss")
    pred_loss_end_time = time.time()
    one_pred_loss_time = pred_loss_end_time - pred_loss_start_time
    total_pred_loss_time += one_pred_loss_time
    if i > 0:
        total_pred_loss_after_compiling_time += one_pred_loss_time

print("Compute Pred Loss Time:", one_pred_loss_time, " sec.")
print("Compute Pred Loss (Ignore Compiling) Time:", total_pred_loss_after_compiling_time, " sec.")

# # 計算總共花費的時間
execution_time = one_var_time + total_pred_loss_time
execution_ignore_compiling_time = one_var_after_compiling_time + total_pred_loss_after_compiling_time
print(f"End2End Compute Time :{execution_time} Sec")
print(f"End2End Compute (Ignore Compiling) Time :{execution_ignore_compiling_time} Sec")
