import csv
import collections
import matplotlib.pyplot as plt
import os
import numpy as np

# M13 with different dropout dense
path_list = [
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/CNN-ResBlock-3-syn-even-skip-augment-funcs-3-std10-phase-shift-2-scale-05-lre4-l2reg1e3-64batch-dropdense0005075-shuffle_2020_11_04_13_25_53/CNN-ResBlock-3-syn-even-skip-augment-funcs-3-std10-phase-shift-2-scale-05-lre4-l2reg1e3-64batch-dropdense0005075-shuffle_2020_11_04_13_25_53_fit_log.csv',
    '',
]
label_list = [
    'L2 1e-3, dropdense 00 05 075',
    ''
]
# M13 with different reg
path_list = [
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/CNN-ResBlock-3-syn-even-skip-augment-funcs-3-lre4-l2e2-128batch_2020_11_06_12_45_49/CNN-ResBlock-3-syn-even-skip-augment-funcs-3-lre4-l2e2-128batch_2020_11_06_12_45_49_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/CNN-ResBlock-3-syn-even-skip-augment-funcs-3-lre4-l2e1_2020_11_06_11_46_15/CNN-ResBlock-3-syn-even-skip-augment-funcs-3-lre4-l2e1_2020_11_06_11_46_15_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/CNN-ResBlock-3-syn-even-skip-merged_train_404298_segments-val-accuracy-lre4_2020_10_14_12_34_33/CNN-ResBlock-3-syn-even-skip-merged_train_404298_segments-val-accuracy-lre4_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/CNN-ResBlock-3-syn-even-skip-merged_train_404298_segments-val-accuracy-lre4-l2e2_2020_10_14_16_51_24/CNN-ResBlock-3-syn-even-skip-merged_train_404298_segments-val-accuracy-lre4-l2e2_fot_history.csv',
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/CNN-ResBlock-3-noskip-syn-even-skip-augment-funcs-3-lre4-l2e2-128batch_2020_11_07_09_18_31/CNN-ResBlock-3-noskip-syn-even-skip-augment-funcs-3-lre4-l2e2-128batch_2020_11_07_09_18_31_fit_log.csv',

]
label_list = [
    # 'Baseline with M13 Data',
    'M13  L2 1e-2 with skip',
    # 'M13 rerun, L2 1e-1',
    # 'M13, L2 1e-3',
    # 'M13, L2 1e-2 original run',
    'M13 , L2 1e-2 no skip'
]
path_list = [
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/CNN-ResBlock-3-syn-even-augment-funcs-3-phase-shift-2-lr1e5-l21e2-128batch-run2_2020_11_14_17_13_27/CNN-ResBlock-3-syn-even-augment-funcs-3-phase-shift-2-lr1e5-l21e2-128batch-run2_2020_11_14_17_13_27_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/CNN-ResBlock-3-syn-even-augment-funcs-3-phase-shift-2-lr1e5-l21e2-AdamAmsGrad-eps1e3_2020_11_12_09_24_48/CNN-ResBlock-3-syn-even-augment-funcs-3-phase-shift-2-lr1e5-l21e2-AdamAmsGrad-eps1e3_2020_11_12_09_24_48_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/CNN-ResBlock-3-syn-even-skip-merged_train_404298_segments-val-accuracy-lre4_2020_10_14_12_34_33/CNN-ResBlock-3-syn-even-skip-merged_train_404298_segments-val-accuracy-lre4_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/CNN-ResBlock-3-syn-even-skip-merged_train_404298_segments-val-accuracy-lre4-l2e2_2020_10_14_16_51_24/CNN-ResBlock-3-syn-even-skip-merged_train_404298_segments-val-accuracy-lre4-l2e2_fot_history.csv',
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-skip-augment-funcs-3-std10-noise-phase-shift-2-scale-05-lre4-l2reg1e3-128batch-shuffle_2020_11_05_11_12_42/cnn-baseline-syn-even-skip-augment-funcs-3-std10-noise-phase-shift-2-scale-05-lre4-l2reg1e3-128batch-shuffle_2020_11_05_11_12_42_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-skip-lre5-l2e2-180batch-no-augment_2020_10_31_16_18_02/cnn-baseline-syn-even-skip-lre5-l2e2-180batch-no-augment_2020_10_31_16_18_02_fit_history.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/CNN-ResBlock-3-noskip-syn-even-skip-augment-funcs-3-lre4-l2e2-128batch_2020_11_07_09_18_31/CNN-ResBlock-3-noskip-syn-even-skip-augment-funcs-3-lre4-l2e2-128batch_2020_11_07_09_18_31_fit_log.csv',
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/CNN-ResBlock-3-syn-even-skip-augment_funcs-3-phase_shift-2-lre4-l2e2-128batch-lr_scheduler_mod10_2020_11_07_23_30_25/CNN-ResBlock-3-syn-even-skip-augment_funcs-3-phase_shift-2-lre4-l2e2-128batch-lr_scheduler_mod10_2020_11_07_23_30_25_fit_log.csv',
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/CNN-ResBlock-3-syn-even-skip-augment_funcs-3-timestep_delta_4-phase_shift-2-lr1e5-l2e2-128batch-lr_Scheduler_10_epochs_factor_01_2020_11_10_09_25_32/CNN-ResBlock-3-syn-even-skip-augment_funcs-3-timestep_delta_4-phase_shift-2-lr1e5-l2e2-128batch-lr_Scheduler_10_epochs_factor_01_2020_11_10_09_25_32_fit_log.csv',
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/CNN-ResBlock-3-syn-even-augment-funcs-3-phase-shift-2-lr1e5-l21e2-128batch-valid_from_train_2020_11_14_19_11_12/CNN-ResBlock-3-syn-even-augment-funcs-3-phase-shift-2-lr1e5-l21e2-128batch-valid_from_train_2020_11_14_19_11_12_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/CNN-ResBlock-3-syn-even-skip-augment_funcs-3-phase_shift-2-lre4-l2e2-128batch-timestep_delta-10_2020_11_07_23_23_03/CNN-ResBlock-3-syn-even-skip-augment_funcs-3-phase_shift-2-lre4-l2e2-128batch-timestep_delta-10_2020_11_07_23_23_03_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/CNN-ResBlock-3-syn-even-skip-augment_funcs-3-phase_shift-2-lre4-l2e2-128batch-timestep_delta-10_2020_11_07_23_23_03/CNN-ResBlock-3-syn-even-skip-augment_funcs-3-phase_shift-2-lre4-l2e2-128batch-timestep_delta-10_2020_11_07_23_23_03_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/CNN-ResBlock-3-syn-even-Highway-augment_funcs-3-timestep_delta_10-phase_shift-2-lre5-l2e2-128batch-lr_scheduler_every_10_2020_11_08_19_38_17/CNN-ResBlock-3-syn-even-Highway-augment_funcs-3-timestep_delta_10-phase_shift-2-lre5-l2e2-128batch-lr_scheduler_every_10_2020_11_08_19_38_17_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/CNN-ResBlock-3-syn-even-Highway-augment_funcs-3-timestep_delta_10-phase_shift-2-lre4-l2e2-128batch-lr_scheduler_every_10_2020_11_08_17_49_22/CNN-ResBlock-3-syn-even-Highway-augment_funcs-3-timestep_delta_10-phase_shift-2-lre4-l2e2-128batch-lr_scheduler_every_10_2020_11_08_17_49_22_fit_log.csv'
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/CNN-ResBlock-syn-even-skip-augment-funcs-2-blackman_nutall-std10-no_horiz_flip-timestep_delta_10-phase-shift-2-scale-05-lre4-l2reg1e2-128batch_2020_11_05_19_36_24/CNN-ResBlock-syn-even-skip-augment-funcs-2-blackman_nutall-std10-no_horiz_flip-timestep_delta_10-phase-shift-2-scale-05-lre4-l2reg1e2-128batch_2020_11_05_19_36_24_fit_log.csv',
]
label_list = [
    # 'Schedule 10, LR 1e-5, Skip, run 2',
    # 'Adam AmsGrad',
    # 'submission model, L2, 1e-3',
    # 'submission model, L2, 1e-2',
    # 'cnn-baseline',
    # 'cnn-baseline 15K Data',
    # 'M13, L2 1e-2, lr 1e-4 , NO SKIP',
    'Schedule 10, LR 1e-5, Skip',
    'Schedule 10, LR 1e-5, Skip, run 3',
    'Schedule 10, LR 1e-5, Skip, valid_from_train'
    # 'submission model , L2 1e-2, timestep delta = 10',
    # 'Plateau, LR 1e-4, Skip, timestep delta = 10',
    # 'Schedule 10, LR 1e-5, Highway, timestep delta = 10',
    # 'Schedule 10, LR 1e-4, Highway, timestep delta = 10',
    # 'Augment funcs x2 and blackman, timestep_delta =10, batch 128'

]

path_list = [
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/CNN-ResBlock-3-syn-even-augment-funcs-3-phase-shift-2-lr1e5-l21e2-128batch-run2_2020_11_14_17_13_27/CNN-ResBlock-3-syn-even-augment-funcs-3-phase-shift-2-lr1e5-l21e2-128batch-run2_2020_11_14_17_13_27_fit_log.csv',
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/CNN-ResBlock-3-syn-even-skip-augment_funcs-3-timestep_delta_4-phase_shift-2-lr1e5-l2e2-128batch-lr_Scheduler_10_epochs_factor_01_2020_11_10_09_25_32/CNN-ResBlock-3-syn-even-skip-augment_funcs-3-timestep_delta_4-phase_shift-2-lr1e5-l2e2-128batch-lr_Scheduler_10_epochs_factor_01_2020_11_10_09_25_32_fit_log.csv',
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/CNN-ResBlock-3-syn-even-augment-funcs-3-phase-shift-2-lr1e5-l21e2-128batch-run4_2020_11_15_19_02_16/CNN-ResBlock-3-syn-even-augment-funcs-3-phase-shift-2-lr1e5-l21e2-128batch-run4_2020_11_15_19_02_16_fit_log.csv',
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/CNN-ResBlock-3-syn-even-augment-funcs-3-phase-shift-2-lr1e5-l21e2-128batch-run5_2020_11_15_20_06_35/CNN-ResBlock-3-syn-even-augment-funcs-3-phase-shift-2-lr1e5-l21e2-128batch-run5_2020_11_15_20_06_35_fit_log.csv',
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/CNN-ResBlock-3-syn-even-augment-funcs-3-phase-shift-2-lr1e5-l21e2-64batch_double-lr_scheduler-run6_2020_11_16_10_30_20/CNN-ResBlock-3-syn-even-augment-funcs-3-phase-shift-2-lr1e5-l21e2-64batch_double-lr_scheduler-run6_2020_11_16_10_30_20_fit_log.csv',
]
label_list = [
    'Schedule 10, LR 1e-5, Skip',
    'run2',
    'run3',
    'run4',
    'run5',
    'run6'

]
path_list = [
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-with_augment-lr1e5-l21e2-128batch_2020_11_17_21_25_19/cnn-baseline-syn-even-with_augment-lr1e5-l21e2-128batch_2020_11_17_21_25_19_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-no_augment-lr1e5-l21e2-128batch_2020_11_17_10_59_35/cnn-baseline-syn-even-no_augment-lr1e5-l21e2-128batch_2020_11_17_10_59_35_fit_log.csv',
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-with_augment-lr1e5-l21e2-128batch-200epochs_2020_11_17_23_18_56/cnn-baseline-syn-even-with_augment-lr1e5-l21e2-128batch-200epochs_2020_11_17_23_18_56_fit_log.csv',
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-no_augment-lr1e5-l21e2-128batch-200epochs-26repeat_numpy_2020_11_17_23_41_27/cnn-baseline-syn-even-no_augment-lr1e5-l21e2-128batch-200epochs-26repeat_numpy_2020_11_17_23_41_27_fit_log.csv',
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/CNN-ResBlock-3-syn-even-skip-augment_funcs-3-phase_shift-2-lre4-l2e2-128batch-lr_scheduler_mod10_2020_11_07_23_30_25/CNN-ResBlock-3-syn-even-skip-augment_funcs-3-phase_shift-2-lre4-l2e2-128batch-lr_scheduler_mod10_2020_11_07_23_30_25_fit_log.csv',
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-augment_funcs_3-phase_shift_2-lr1e5-l21e2-128batch-dropout_after_conv2d_05_2020_11_19_10_50_54/cnn-baseline-syn-even-augment_funcs_3-phase_shift_2-lr1e5-l21e2-128batch-dropout_after_conv2d_05_2020_11_19_10_50_54_fit_log.csv',
]
label_list = [
    'baseline with augmentation',
    'baseline without augmentation',
    'proposed Model',
    'baseline with augmentation and do'
]
# # CNN- RESBLOCK lr sweep
# path_list = [
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/Sweep-CNN-ResBlock-syn-even-no_augment-lr1e5-l21e2-128batch-lr_sweep_2020_11_17_21_34_42/Sweep-CNN-ResBlock-syn-even-no_augment-lr1e5-l21e2-128batch-lr_sweep_2020_11_17_21_34_42_learning_rate_1e-05_fit_log.csv',
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/Sweep-CNN-ResBlock-syn-even-no_augment-lr1e5-l21e2-128batch-lr_sweep_2020_11_17_21_34_42/Sweep-CNN-ResBlock-syn-even-no_augment-lr1e5-l21e2-128batch-lr_sweep_2020_11_17_21_34_42_learning_rate_0.0001_fit_log.csv',
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/Sweep-CNN-ResBlock-syn-even-no_augment-lr1e5-l21e2-128batch-lr_sweep_2020_11_17_21_34_42/Sweep-CNN-ResBlock-syn-even-no_augment-lr1e5-l21e2-128batch-lr_sweep_2020_11_17_21_34_42_learning_rate_0.001_fit_log.csv',
# ]
# label_list = [
#     '1e-5',
#     '1e-4',
#     '1e-3',
# ]
# # CNN- RESBLOCK reg term sweep
# path_list = [
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/Sweep-CNN-ResBlock-syn-even-no_augment-lr1e5-l21e2-128batch-reg_term_sweep_2020_11_17_21_44_18/Sweep-CNN-ResBlock-syn-even-no_augment-lr1e5-l21e2-128batch-reg_term_sweep_2020_11_17_21_44_18_Regularization_term_0.0001_fit_log.csv',
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/Sweep-CNN-ResBlock-syn-even-no_augment-lr1e5-l21e2-128batch-reg_term_sweep_2020_11_17_21_44_18/Sweep-CNN-ResBlock-syn-even-no_augment-lr1e5-l21e2-128batch-reg_term_sweep_2020_11_17_21_44_18_Regularization_term_0.001_fit_log.csv',
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/Sweep-CNN-ResBlock-syn-even-no_augment-lr1e5-l21e2-128batch-reg_term_sweep_2020_11_17_21_44_18/Sweep-CNN-ResBlock-syn-even-no_augment-lr1e5-l21e2-128batch-reg_term_sweep_2020_11_17_21_44_18_Regularization_term_0.01_fit_log.csv',
# ]
# label_list = [
#     '1e-4',
#     '1e-3',
#     'e1-2',
# ]


def csv_to_dict(csv_data):
    is_first_line = True
    csv_dict = collections.OrderedDict()
    for line in csv_data:
        if is_first_line:
            is_first_line = False
            for key in line:
                csv_dict[key] = []
        else:
            for key, data in zip(csv_dict.keys(), line):
                csv_dict[key].append(data)
    return csv_dict


def plot_metric(metric, fit_history_list, start_epoch=0, mode="max"):
    if mode == "min":
        final_epoch = min([len(fit_dict['loss']) for fit_dict in fit_history_list])
        # final_epoch += 5
    else:
        final_epoch = max([len(fit_dict['loss']) for fit_dict in fit_history_list])
    plt.figure()
    # train
    for i, fit_dict in enumerate(fit_history_list):
        epochs = fit_dict['epoch'][start_epoch:final_epoch]
        # train loss
        label_train = 'Model {} train'.format(i + 1)
        label_train = '{}'.format(label_list[i])
        plt.plot(epochs, fit_dict['{}'.format(metric)][start_epoch:final_epoch], linestyle='-', lw=2,
                 label=label_train)
        color = plt.gca().lines[-1].get_color()
        # validation loss
        label_valid = 'Model {} valid'.format(i + 1)
        plt.plot(epochs, fit_dict['val_{}'.format(metric)][start_epoch:final_epoch],linestyle='--', lw=2,
                 color=color)
        val_values = fit_dict['val_{}'.format(metric)][start_epoch:final_epoch]
        if i == 2 and metric == 'auc':
            plt.scatter(epochs[-1:],val_values[-1:], s=50 ,color=color, marker='^')
        # if i == 2 and metric == 'auc':
        #     epoch_lr_sched = [e if 0 == e % 10 for e in epochs]
        #     plt.scatter(epoch_lr_sched,val_values[-1:], s=50 ,color=color, marker='^')

    plt.title(metric, fontsize=30)
    plt.grid()
    if start_epoch != 0:
        plt.xticks(np.arange(start_epoch, final_epoch, 25))
    plt.legend(loc="best",fontsize=15)
    plt.xlabel('Epochs', fontsize=20)
    # ylabel = metric if metric = 'loss' and metric != 'val_loss' else
    plt.savefig(os.path.join(fit_history_graph_dir, '{}.png'.format(metric)))


fit_history_list = []
for path in path_list:
    with open(path, 'r') as csv_file:
        fit_csv = csv.reader(csv_file, delimiter=',', quotechar='|')
        csv_dict = csv_to_dict(fit_csv)
        for key in csv_dict.keys():
            if key == 'epoch':
                csv_dict[key] = [int(x) for x in csv_dict[key]]
            else:
                csv_dict[key] = [float(x) for x in csv_dict[key]]
        fit_history_list.append(csv_dict)

fit_history_graph_dir = '{}/../fit_history_dir'.format(os.getcwd())
if not os.path.exists(fit_history_graph_dir):
    os.makedirs(fit_history_graph_dir)

# for fit_dict in fit_history_list:
#     fit_dict['loss'] = np.log10(fit_dict['loss'])
#     fit_dict['val_loss'] = np.log10(fit_dict['val_loss'])

plot_metric('accuracy', fit_history_list)
plot_metric('loss', fit_history_list,start_epoch=5)
plot_metric('auc', fit_history_list)

print('')

#
# DELTA_INTERVAL = 100
# _ = np.mean([x_aug - x for x_aug,x in zip(fit_history_list[0]['val_auc'][-20:],fit_history_list[1]['val_auc'][-20:])])
# print(_)
