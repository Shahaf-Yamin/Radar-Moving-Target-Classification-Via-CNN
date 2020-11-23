from utils.utils import preprocess_meta_data
from tensorflow.keras.layers import LeakyReLU
from data.data_parser import *

def kernel_init_sim():
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

def split_public_test_to_valid():
    SRC_DIR = os.getcwd()
    RADAR_DIR = os.path.join(SRC_DIR, os.pardir)
    config = preprocess_meta_data(SRC_DIR)
    exp_name = config.exp_name
    data_parser = DataSetParser(stable_mode=False, config=config, read_validation_only=True)
    test_data, train_data = data_parser.split_public_test_valid()
    print('saving FULL PUBLIC TRAIN AND TEST IN DIR: {}'.format(os.getcwd()))
    np.save('FULL_PUBLIC_TRAIN',train_data)
    np.save('FULL_PUBLIC_TEST',test_data)


if __name__ == '__main__':
    print('Current working directory is: {}'.format(os.getcwd()))
    # kernel_init_sim()
    split_public_test_to_valid()
