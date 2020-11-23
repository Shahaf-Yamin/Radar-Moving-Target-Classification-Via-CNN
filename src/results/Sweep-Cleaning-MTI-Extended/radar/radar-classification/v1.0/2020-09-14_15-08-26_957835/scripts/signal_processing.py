import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from filterpy.kalman import KalmanFilter
import numpy as np
import os
#
def eClean_algorithm(iq_mat):
    #Calculate histogram
    histogram, bin_edges = np.histogram(iq_mat, bins=32*126, range=(np.min(iq_mat), np.max(iq_mat)))
    first_derviative = np.diff(histogram, n=1)

    last_index = len(first_derviative) - 2
    threshold = 0.15

    while( np.mean(first_derviative[last_index-40:last_index]) <= threshold ):
        last_index = last_index - 1

    mask = iq_mat < np.linspace(np.min(iq_mat), np.max(iq_mat), int(histogram.shape[0]))[last_index]
    iq_mat[mask] = 0

    return iq_mat

# def exponential_average_background_subtraction(iq_mat):
#     y_k = np.copy(iq_mat[:,0])
#     alpha = 0.05
#     for slow_axis_index in range(1, iq_mat.shape[1]):
#         y_k = alpha * y_k + (1 - alpha) * iq_mat[:, slow_axis_index]

def kalman_filter_clean_algorithm(iq_mat):
    plt.figure()
    plt.imshow(iq_mat)
    plt.savefig("before_kalman_filter")

    #
    # Filter = KalmanFilter(dim_x=1, dim_z=iq_mat.shape[0])
    # Filter.x = 0
    #process_uncertaintiy = np.ones((1, iq_mat.shape[0]))[0]
    #measurment_uncertaintiy = np.ones((1, iq_mat.shape[0]))[0]
    # main_diag = np.ones((1,iq_mat.shape[0]))[0]


    #Filter.R = np.diag(measurment_uncertaintiy,0)

    iq_org = np.copy(iq_mat)
    Filter = KalmanFilter(dim_x=iq_mat.shape[0], dim_z=iq_mat.shape[0])
    Filter.H = np.ones((1, iq_mat.shape[0]))[0]
    Filter.x = np.zeros((iq_mat.shape[0],1))
    Filter.update(z=np.copy(iq_mat[:, 0]))
    for slow_axis_index in range(1, iq_mat.shape[1]):
        Filter.predict()
        Filter.update(z=np.copy(iq_mat[:, slow_axis_index]))
        iq_mat[:, slow_axis_index] = np.copy(Filter.x[:,0])


        # for fast_axis_index in range(1, iq_mat.shape[0]):
        # # plt.figure()
        # # plt.plot(Filter.x)
        # # plt.savefig("kalman_filter_estimation")
        #     Filter.predict()
        #     Filter.update(z=np.copy(iq_mat[fast_axis_index,slow_axis_index]))
        # iq_mat[:, slow_axis_index] = iq_mat[:, slow_axis_index] - Filter.x

    plt.figure()
    plt.plot(iq_mat[:,1])
    plt.plot(iq_org[:,1])
    plt.savefig('compare_Kalman')

    plt.figure()
    plt.imshow(iq_mat)
    plt.savefig("after_kalman_filter")
    return iq_mat




def plot_results(IMFs):
    plt.figure(figsize=(16, 8))
    plt.plot(range(IMFs[0]), IMFs[0], '0')
    plt.grid(True)
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (secs)')
    plt.show()
    plt.figure(figsize=(16, 8))
    plt.grid(True)
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (secs)')
    plt.show()
    plt.plot(range(IMFs[1]), IMFs[1], '1')
    plt.figure(figsize=(16, 8))
    plt.grid(True)
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (secs)')
    plt.show()
    plt.plot(range(IMFs[2]), IMFs[2], '2')
    plt.grid(True)
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (secs)')
    plt.show()
    plt.figure(figsize=(16, 8))
    plt.plot(range(IMFs[3]), IMFs[3], '3')
    plt.grid(True)
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (secs)')
    plt.show()