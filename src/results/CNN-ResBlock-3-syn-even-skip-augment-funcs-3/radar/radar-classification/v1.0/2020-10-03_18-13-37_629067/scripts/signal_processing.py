import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from filterpy.kalman import KalmanFilter
import numpy as np
import os


#
def eClean_algorithm(X):
    # Calculate histogram
    for iq_mat in X:
        histogram, bin_edges = np.histogram(iq_mat, bins=32 * 126, range=(np.min(iq_mat), np.max(iq_mat)))
        first_derviative = np.diff(histogram, n=1)

        last_index = len(first_derviative) - 2
        threshold = 0.15

        while (np.mean(first_derviative[last_index - 40:last_index]) <= threshold):
            last_index = last_index - 1

        mask = iq_mat < np.linspace(np.min(iq_mat), np.max(iq_mat), int(histogram.shape[0]))[last_index]
        iq_mat[mask] = 0
    return X


# def exponential_average_background_subtraction(iq_mat):
#     y_k = np.copy(iq_mat[:,0])
#     alpha = 0.05
#     for slow_axis_index in range(1, iq_mat.shape[1]):
#         y_k = alpha * y_k + (1 - alpha) * iq_mat[:, slow_axis_index]

def kalman_filter_clean_algorithm(X):
    # plt.figure()
    # plt.imshow(iq_mat)
    # plt.savefig("before_kalman_filter")

    #
    # Filter = KalmanFilter(dim_x=1, dim_z=iq_mat.shape[0])
    # Filter.x = 0
    # process_uncertaintiy = np.ones((1, iq_mat.shape[0]))[0]
    # measurment_uncertaintiy = np.ones((1, iq_mat.shape[0]))[0]
    # main_diag = np.ones((1,iq_mat.shape[0]))[0]

    # Filter.R = np.diag(measurment_uncertaintiy,0)
    for iq_mat in X:
        iq_org = np.copy(iq_mat)
        Filter = KalmanFilter(dim_x=iq_mat.shape[0], dim_z=iq_mat.shape[0])
        Filter.H = np.ones((1, iq_mat.shape[0]))[0]
        Filter.x = np.zeros((iq_mat.shape[0], 1))
        Filter.update(z=np.copy(iq_mat[:, 0]))
        for slow_axis_index in range(1, iq_mat.shape[1]):
            Filter.predict()
            Filter.update(z=np.copy(iq_mat[:, slow_axis_index]))
            iq_mat[:, slow_axis_index] = np.copy(Filter.x[:, 0])

            # for fast_axis_index in range(1, iq_mat.shape[0]):
            # # plt.figure()
            # # plt.plot(Filter.x)
            # # plt.savefig("kalman_filter_estimation")
            #     Filter.predict()
            #     Filter.update(z=np.copy(iq_mat[fast_axis_index,slow_axis_index]))
            # iq_mat[:, slow_axis_index] = iq_mat[:, slow_axis_index] - Filter.x

    # plt.figure()
    # plt.plot(iq_mat[:,1])
    # plt.plot(iq_org[:,1])
    # plt.savefig('compare_Kalman')
    #
    # plt.figure()
    # plt.imshow(iq_mat)
    # plt.savefig("after_kalman_filter")
    return X


def PCA_expansion(iq_mat, config):
    '''
    Calculate the PCA of the vector and generate samples from it
    '''

    standardized_iq_mat = iq_mat - np.mean(iq_mat, axis=0)
    cov_mat = np.cov(standardized_iq_mat)  # Calculate the covariance matrix

    '''
    Eigen Value decomposition
    '''
    eigen_values, eigen_vector = np.linalg.eigh(cov_mat)  # Calculate the eigen values and eigen vectors
    idx = np.argsort(eigen_values)[::-1]
    eigen_vector = eigen_vector[:, idx]

    '''
    Generate new samples
    '''
    NUMBER_OF_KL_COEEFIECENT = config.number_of_pca_coeff
    VARIANCE_SCALING = config.pca_augmentation_scaling

    KL_mat = np.matmul(np.transpose(np.conjugate(standardized_iq_mat)), eigen_vector)
    KL_mat[:, 5:NUMBER_OF_KL_COEEFIECENT] = KL_mat[:, :NUMBER_OF_KL_COEEFIECENT] + VARIANCE_SCALING * (
        np.multiply(KL_mat[:, :NUMBER_OF_KL_COEEFIECENT], np.random.randn(32, NUMBER_OF_KL_COEEFIECENT)))

    new_iq_mat = np.matmul(eigen_vector, np.transpose(np.conjugate(KL_mat)))

    figure, axes = plt.subplots(nrows=2, ncols=2)
    axes[0, 0].imshow(standardized_iq_mat)
    axes[0, 0].set_title('Freq Response Org')
    axes[0, 1].imshow(new_iq_mat)
    axes[0, 1].set_title('Freq Response Generated 1 ')
    axes[1, 0].imshow(new_iq_mat)
    axes[1, 0].set_title('Freq Response Generated 2 ')
    axes[1, 1].imshow(new_iq_mat)
    axes[1, 1].set_title('Freq Response Generated 3 ')

    figure.savefig('test_pca')

    return new_iq_mat


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


def Sample_rectangle_from_spectrogram(iq_mat, config):
    NUM_OF_TIMESTEPS = config.rect_augment_num_of_timesteps
    MAX_NUM_OF_TIMESTEPS = 32
    new_iq_list = []

    for col_ind in range(MAX_NUM_OF_TIMESTEPS - NUM_OF_TIMESTEPS):
        new_iq_list.append(iq_mat[:, col_ind:col_ind + NUM_OF_TIMESTEPS])

    figure, axes = plt.subplots(nrows=2, ncols=2)
    # axes[0, 0].imshow(iq_mat)
    # axes[0, 0].set_title('Freq Response Org')
    # index = np.random.choice(range(len(new_iq_list)))
    # axes[0, 1].imshow(new_iq_list[index])
    # axes[0, 1].set_title('Freq Response Extracted index'.format(index))
    # index = np.random.choice(range(len(new_iq_list)))
    # axes[1, 0].imshow(new_iq_list[index])
    # axes[1, 0].set_title('Freq Response Extracted {}'.format(index))
    # index = np.random.choice(range(len(new_iq_list)))
    # axes[1, 1].imshow(new_iq_list[index])
    # axes[1, 1].set_title('Freq Response Extracted {} '.format(index))
    #
    # figure.savefig('test_rect_sample')

    return new_iq_list


def ZCA_transform(iq):
    org_iq = np.copy(iq)
    X = iq.reshape(iq.shape[0],-1)
    # A = X.reshape(org_iq.shape)
    # Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
    sigma = np.cov(X, rowvar=True)  # [M x M]
    # Singular Value Decomposition. X = U * np.diag(S) * V
    U, S, V = np.linalg.svd(sigma)
    # U: [M x M] eigenvectors of sigma.
    # S: [M x 1] eigenvalues of sigma.
    # V: [M x M] transpose of U
    epsilon = 1e-5
    # ZCA Whitening matrix: U * Lambda * U'
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T))  # [M x M]
    Y = np.dot(ZCAMatrix, X)
    Y = Y.reshape(org_iq.shape)
    # # center the data
    # real_vec = iq[:,:,0].ravel()
    # imag_vec = iq[:,:,1].ravel()
    # # X = np.vstack((real_vec, imag_vec))
    # X = np.vstack((real_vec, imag_vec))
    # # A = (X.swapaxes(0, 1)).reshape(org_iq.shape)
    # # assert A.all() == org_iq.all()
    #
    # X = X - np.expand_dims(np.mean(X, axis=1), axis=-1)
    # n = X.shape[1]
    # # correlation matrix
    # # Rx = np.cov(X)
    # Rx = (1 / n - 1) * np.matmul(X , X.transpose())
    # # validate real symmetric
    # assert np.linalg.norm(Rx.transpose() - Rx) == 0
    #
    # # EVD for Rx
    # eigen_values, P = np.linalg.eigh(Rx)
    # # sqrt can't be taken on negative values
    # neg_indices = np.where(eigen_values < 0)
    # eigen_values[neg_indices] = -1.0 * eigen_values[neg_indices]
    # P[:, neg_indices] = -1.0 * P[:, neg_indices]
    # assert np.where(eigen_values < 0)[0].tolist() == []  # validate no negative in eigen_values
    # D = np.diag(eigen_values)
    # # Whitening filter, W = ((n-1)^(1/2))* P * (D^(-1/2)) * P.T
    # i, j = np.indices(D.shape)
    # D[i == j] = np.float_power(D[i == j], -0.5)
    # W = (np.sqrt(n-1)) * np.matmul(np.matmul(P, D), P.transpose())
    # # Y = W*X
    # Y = np.matmul(W, X)
    #
    # Y = (Y.swapaxes(0,1)).reshape(org_iq.shape)
    # # Y = Y.reshape(org_iq.shape)
    return Y
