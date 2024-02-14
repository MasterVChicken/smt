"""
Predicting Airfoil Aerodynamics through data by Raul Carreira Rufato and Prof. Joseph Morlier
"""

# Gradient Enhanced Neural Networks

import numpy as np
import zfpy
import os


from smt.examples.airfoil_parameters.learning_airfoil_parameters import (
    load_cd_training_data,
    load_NACA4412_modeshapes,
    plot_predictions,
)
from sklearn.model_selection import train_test_split
from smt.surrogate_models.genn import GENN, load_smt_data

def compress_and_calculate_ratio(data):
    # Compress data
    compressed_data = zfpy.compress_numpy(data, tolerance=1e-4)

    # Caculate data size before and after compression
    original_size = data.nbytes
    compressed_size = len(compressed_data)

    # caculate compression ratio
    compression_ratio = original_size / compressed_size

    return compression_ratio, original_size, compressed_size

x, y, dy = load_cd_training_data()
WORKDIR = os.path.dirname(os.path.abspath(__file__))

# splitting the dataset
x_train, x_test, y_train, y_test, dy_train, dy_test = train_test_split(
    x, y, dy, train_size=0.8
)
# building and training the GENN
genn = GENN(print_global=False)
# compression
genn.options["comp"] = True
# learning rate that controls optimizer step size
genn.options["alpha"] = 0.001
# lambd = 0. = no regularization, lambd > 0 = regularization
genn.options["lambd"] = 0.1
# gamma = 0. = no grad-enhancement, gamma > 0 = grad-enhancement
genn.options["gamma"] = 1.0
# number of hidden layers
genn.options["deep"] = 2
# number of nodes per hidden layer
genn.options["wide"] = 6
# used to divide data into training batches (use for large data sets)
genn.options["mini_batch_size"] = 256
# number of passes through data
genn.options["num_epochs"] = 5
# number of optimizer iterations per mini-batch
genn.options["num_iterations"] = 10
# print output (or not)
genn.options["is_print"] = False

# convenience function to read in data that is in SMT format
# load_smt_data(genn, x_train, y_train, dy_train)

# 将load_data模组改为内置函数
genn.load_smt_data(x_train, y_train, dy_train)

genn.train()

# 其实没有太大用处，主要是为了展现训练过程中的convergence and accuracy指标
## non-API function to plot training history (to check convergence)
genn.plot_training_history()
## non-API function to check accuracy of regression
genn.goodness_of_fit(x_test, y_test, dy_test)

# API function to predict values at new (unseen) points
y_pred = genn.predict_values(x_test)

# Now we will use the trained model to make a prediction with a not-learned form.
# Example Prediction for NACA4412.
# Airfoil mode shapes should be determined according to Bouhlel, M.A., He, S., and Martins,
# J.R.R.A., mSANN Model Benchmarks, Mendeley Data, 2019. https://doi.org/10.17632/ngpd634smf.1
# Comparison of results with Adflow software for an alpha range from -1 to 7 degrees. Re = 3000000
airfoil_modeshapes = load_NACA4412_modeshapes()
Ma = 0.3
alpha = 0

# input in neural network is created out of airfoil mode shapes, Mach number and alpha
# airfoil_modeshapes: computed mode_shapes of random airfol geometry with parameterise_airfoil
# Ma: desired Mach number for evaluation in range [0.3,0.6]
# alpha: scalar in range [-1, 6]
input = np.zeros(shape=(1, 16))
input[0, :14] = airfoil_modeshapes
input[0, 14] = Ma
input[0, -1] = alpha

# prediction
cd_pred = genn.predict_values(input)
print("Drag coefficient prediction (cd): ", cd_pred[0, 0])

# plot_predictions(airfoil_modeshapes, Ma, genn)
genn.plot_predictions_with_compression(airfoil_modeshapes, Ma, WORKDIR)

x_comp_ratio, x_size, x_comp_size = compress_and_calculate_ratio(x)
y_comp_ratio, y_size, y_comp_size = compress_and_calculate_ratio(y)
dy_comp_ratio, dy_size, dy_comp_size = compress_and_calculate_ratio(dy)
print(x_comp_ratio, x_size, x_comp_size)
print(y_comp_ratio, y_size, y_comp_size)
print(dy_comp_ratio, dy_size, dy_comp_size)