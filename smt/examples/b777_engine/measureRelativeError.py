import numpy as np

from smt.surrogate_models import RMTB
from smt.examples.b777_engine.b777_engine import get_b777_engine, plot_b777_engine, get_b777_engine_compressed, plot_b777_engine_with_compression, compress_and_calculate_ratio

xt, yt, dyt_dxt, xlimits = get_b777_engine()

interp = RMTB(
    num_ctrl_pts=15,
    xlimits=xlimits,
    nonlinear_maxiter=20,
    approx_order=2,
    energy_weight=0e-14,
    regularization_weight=0e-18,
    extrapolate=True,
)
interp.set_training_values(xt, yt)
interp.set_training_derivatives(xt, dyt_dxt[:, :, 0], 0)
interp.set_training_derivatives(xt, dyt_dxt[:, :, 1], 1)
interp.set_training_derivatives(xt, dyt_dxt[:, :, 2], 2)
interp.train()

# plot_b777_engine(xt, yt, xlimits, interp)

xt_c, yt_c, dyt_dxt_c, xlimits_c = get_b777_engine_compressed()

interp_c = RMTB(
    num_ctrl_pts=15,
    xlimits=xlimits_c,
    nonlinear_maxiter=20,
    approx_order=2,
    energy_weight=0e-14,
    regularization_weight=0e-18,
    extrapolate=True,
)
interp_c.set_training_values(xt_c, yt_c)
interp_c.set_training_derivatives(xt_c, dyt_dxt_c[:, :, 0], 0)
interp_c.set_training_derivatives(xt_c, dyt_dxt_c[:, :, 1], 1)
interp_c.set_training_derivatives(xt_c, dyt_dxt_c[:, :, 2], 2)
interp_c.train()
# plot_b777_engine(xt_c, yt_c, xlimits_c, interp_c)

num_rows_to_select = 100
relative_acc_SFC = np.zeros(100)
relative_acc_thrust = np.zeros(100)
i = 0
random_indices = np.random.choice(xt.shape[0], size=num_rows_to_select, replace=False)
print(random_indices)
selected_rows = xt[random_indices, :]
selected_results = yt[random_indices, :]
# print(selected_rows.shape)
# print(selected_rows)

for index in selected_rows:
    input_t = index.reshape(1, 3)
    predict_value = interp_c.predict_values(input_t)
    actual_value = selected_results[i]

    relative_acc_SFC[i] = 1-abs(predict_value[0][1] - actual_value[1]) / actual_value[1]
    relative_acc_thrust[i] = 1-abs(predict_value[0][0] - actual_value[0]) / actual_value[0]

    i = i + 1

print("This is relative acc of SFC\n")
print(relative_acc_SFC)
print("This is relative acc of thrust\n")
print(relative_acc_thrust)
print("Final acc of SFC")
print(np.sum(relative_acc_SFC) / 100)
print("Final acc of thrust")
print(np.sum(relative_acc_thrust) / 100)

plot_b777_engine_with_compression(xt, yt, xlimits, interp, interp_c)

xt_comp_ratio, xt_size, xt_comp_size = compress_and_calculate_ratio(xt)
yt_comp_ratio, yt_size, yt_comp_size = compress_and_calculate_ratio(yt)
dyt_dxt_comp_ratio, dyt_dxt_size, dyt_dxt_comp_size = compress_and_calculate_ratio(dyt_dxt)

print(xt_comp_ratio, xt_size, xt_comp_size)
print(yt_comp_ratio, yt_size, yt_comp_size)
print(dyt_dxt_comp_ratio, dyt_dxt_size, dyt_dxt_comp_size)