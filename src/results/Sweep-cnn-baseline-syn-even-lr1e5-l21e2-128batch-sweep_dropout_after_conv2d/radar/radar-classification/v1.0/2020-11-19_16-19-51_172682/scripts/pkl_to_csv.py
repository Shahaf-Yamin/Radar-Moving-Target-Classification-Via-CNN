import pickle
import base64
import csv

FILE_NAME = '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-skip-lre5-l2e2-180batch-phase-shift1_2020_10_31_16_39_36/cnn-baseline-syn-even-skip-lre5-l2e2-180batch-phase-shift1_2020_10_31_16_39_36_fit_history'
PKL_PATH = '{}.pkl'.format(FILE_NAME)
CSV_PATH = '{}.csv'.format(FILE_NAME)

pickle_dict = pickle.loads(open(PKL_PATH, 'rb').read())
with open(CSV_PATH, 'w', encoding='utf8') as csv_file:
    wr = csv.writer(csv_file)
    key_list = list(pickle_dict.keys())
    key_list.insert(0, 'epoch')
    wr.writerow(key_list)
    for i in range(len(pickle_dict[key_list[1]])):
        value_list = [pickle_dict[key][i] for key in key_list[1:]]
        value_list.insert(0, i)
        wr.writerow(value_list)
