import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU
import numpy as np

cum_mean = 0.0
cum_std = 0.0
x = np.random.randn(512)
N = 50
for i in range(N):

    # he normal
    # a = np.random.randn(512,512) * np.sqrt(2./512)
    # a = np.random.randn(512,512) * np.sqrt(2./512)
    # a = np.random.randn(512,512)

    # glorot uniform
    limit = np.sqrt(6. /1024)
    a = np.random.uniform(-limit,limit,(512,512))

    relu = LeakyReLU(alpha=0.1)
    x = relu(np.matmul(a,x))
    cum_mean += np.mean(x)
    cum_std += np.std(x)
print('with glorot uniform')
# print('with he normal')
print('average mean = {}'.format(cum_mean / N))
print('average std = {}'.format(cum_std / N))
print('final mean = {} after {} layers'.format(np.mean(x), N))
print('final std = {} after {} layers'.format(np.std(x), N))