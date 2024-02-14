import numpy

from smt.surrogate_models import RMTB
from smt.examples.b777_engine.b777_engine import get_b777_engine, plot_b777_engine, get_b777_engine_compressed

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
plot_b777_engine()

# 我也没有搞清楚输入输出的含义究竟是什么
# 但是我查了一下github上面smt的issue
# 据说input是throttle, altitude, Mach number
# throttle是油门
# altitude是海拔
# Mach Number是马赫（速度单位）
# outputs是SFC, thrust
# Specific Fuel Consumption(SFC): 单位油耗比
# thrust: 推力比
