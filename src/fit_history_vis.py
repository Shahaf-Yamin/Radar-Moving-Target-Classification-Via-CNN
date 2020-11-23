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

# Data Size, Same Model
path_list = [
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-lr1e5-l21e3_2020_11_21_12_10_25/cnn-baseline-lr1e5-l21e3_2020_11_21_12_10_25_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-lr1e5-l21e3_2020_11_21_12_12_13/cnn-baseline-syn-even-lr1e5-l21e3_2020_11_21_12_12_13_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_2-lr1e5-l21e3_2020_11_21_12_14_42/cnn-baseline-syn-even-phase_shift_2-lr1e5-l21e3_2020_11_21_12_14_42_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_2-vertical_flip-lr1e5-l21e3_2020_11_21_12_20_50/cnn-baseline-syn-even-phase_shift_2-vertical_flip-lr1e5-l21e3_2020_11_21_12_20_50_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_2-vertical_flip-horiz_flip-lr1e5-l21e3_2020_11_21_12_55_38/cnn-baseline-syn-even-phase_shift_2-vertical_flip-horiz_flip-lr1e5-l21e3_2020_11_21_12_55_38_fit_log.csv',
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_2-augment_funcs_1-lr1e5-l21e3_2020_11_21_13_00_32/cnn-baseline-syn-even-phase_shift_2-augment_funcs_1-lr1e5-l21e3_2020_11_21_13_00_32_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_2-augment_funcs_2-lr1e5-l21e3_2020_11_21_13_06_32/cnn-baseline-syn-even-phase_shift_2-augment_funcs_2-lr1e5-l21e3_2020_11_21_13_06_32_fit_log.csv',
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_2-augment_funcs_3-lr1e5-l21e3_2020_11_21_13_07_40/cnn-baseline-syn-even-phase_shift_2-augment_funcs_3-lr1e5-l21e3_2020_11_21_13_07_40_fit_log.csv',
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/CNN-ResBlock-3-syn-even-skip-augment_funcs-3-phase_shift-2-lre4-l2e2-128batch-lr_scheduler_mod10_2020_11_07_23_30_25/CNN-ResBlock-3-syn-even-skip-augment_funcs-3-phase_shift-2-lre4-l2e2-128batch-lr_scheduler_mod10_2020_11_07_23_30_25_fit_log.csv',

]
label_list = [
    # '7K',
    # '15K',
    # '45K',
    # '90K',
    # '135K',
    '225K Data, Baseline + L2',
    # '310K',
    '400K Data, Baseline + L2',
    'proposed Model',
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
path_list = [
    # 1layer
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-1layers-syn-even-augment_funcs_3-phase_shift_2-lr1e5-l21e3-128batch_2020_11_20_19_24_31/cnn-baseline-1layers-syn-even-augment_funcs_3-phase_shift_2-lr1e5-l21e3-128batch_2020_11_20_19_24_31_fit_log.csv',
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-1layer-syn-even-phase_shift_2-augment_funcs_3-lr1e5-l21e3-run2_2020_11_21_18_27_45/cnn-baseline-1layer-syn-even-phase_shift_2-augment_funcs_3-lr1e5-l21e3-run2_2020_11_21_18_27_45_fit_log.csv',
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-1layer-syn-even-phase_shift_2-augment_funcs_3-lr1e5-l21e3-run3_2020_11_21_18_28_58/cnn-baseline-1layer-syn-even-phase_shift_2-augment_funcs_3-lr1e5-l21e3-run3_2020_11_21_18_28_58_fit_log.csv',
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-1layer-syn-even-phase_shift_2-augment_funcs_3-lr1e5-l21e3-run4_2020_11_21_18_32_37/cnn-baseline-1layer-syn-even-phase_shift_2-augment_funcs_3-lr1e5-l21e3-run4_2020_11_21_18_32_37_fit_log.csv',

    # 2layer
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_2-augment_funcs_3-lr1e5-l21e3_2020_11_21_13_07_40/cnn-baseline-syn-even-phase_shift_2-augment_funcs_3-lr1e5-l21e3_2020_11_21_13_07_40_fit_log.csv',
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-2layers-syn-even-augment_funcs_3-phase_shift_2-lr1e5-l21e3-128batch_2020_11_20_19_22_55/cnn-baseline-2layers-syn-even-augment_funcs_3-phase_shift_2-lr1e5-l21e3-128batch_2020_11_20_19_22_55_fit_log.csv',
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-2layer-syn-even-phase_shift_2-augment_funcs_3-lr1e5-l21e3-run3_2020_11_21_18_38_22/cnn-baseline-2layer-syn-even-phase_shift_2-augment_funcs_3-lr1e5-l21e3-run3_2020_11_21_18_38_22_fit_log.csv',
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-2layer-syn-even-phase_shift_2-augment_funcs_3-lr1e5-l21e3-run4_2020_11_21_18_42_11/cnn-baseline-2layer-syn-even-phase_shift_2-augment_funcs_3-lr1e5-l21e3-run4_2020_11_21_18_42_11_fit_log.csv',

    # # 4layer
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-4layers-syn-even-augment_funcs_3-phase_shift_2-lr1e5-l21e3-128batch_2020_11_20_19_34_32/cnn-baseline-4layers-syn-even-augment_funcs_3-phase_shift_2-lr1e5-l21e3-128batch_2020_11_20_19_34_32_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-4layer-syn-even-phase_shift_2-augment_funcs_3-lr1e5-l21e3-run2_2020_11_21_18_46_23/cnn-baseline-4layer-syn-even-phase_shift_2-augment_funcs_3-lr1e5-l21e3-run2_2020_11_21_18_46_23_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-4layers-syn-even-phase_shift_2-augment_funcs_3-lr1e5-l21e3-run3_2020_11_21_20_29_19/cnn-baseline-4layers-syn-even-phase_shift_2-augment_funcs_3-lr1e5-l21e3-run3_2020_11_21_20_29_19_fit_log.csv',
    #
    # # 8layer
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-8layers-syn-even-augment_funcs_3-phase_shift_2-lr1e5-l21e3-128batch_2020_11_20_23_56_33/cnn-baseline-8layers-syn-even-augment_funcs_3-phase_shift_2-lr1e5-l21e3-128batch_2020_11_20_23_56_33_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-8layers-syn-even-augment_funcs_3-phase_shift_2-lr1e5-l21e3-128batch-run2_2020_11_21_23_43_32/cnn-baseline-8layers-syn-even-augment_funcs_3-phase_shift_2-lr1e5-l21e3-128batch-run2_2020_11_21_23_43_32_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-8layers-syn-even-augment_funcs_3-phase_shift_2-lr1e5-l21e3-128batch-run3_2020_11_21_23_44_55/cnn-baseline-8layers-syn-even-augment_funcs_3-phase_shift_2-lr1e5-l21e3-128batch-run3_2020_11_21_23_44_55_fit_log.csv',

    # Proposed Model
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/CNN-ResBlock-3-syn-even-skip-augment_funcs-3-phase_shift-2-lre4-l2e2-128batch-lr_scheduler_mod10_2020_11_07_23_30_25/CNN-ResBlock-3-syn-even-skip-augment_funcs-3-phase_shift-2-lre4-l2e2-128batch-lr_scheduler_mod10_2020_11_07_23_30_25_fit_log.csv',
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/CNN-ResBlock-3-syn-even-augment-funcs-3-phase-shift-2-lr1e5-l21e2-128batch-run2_2020_11_14_17_13_27/CNN-ResBlock-3-syn-even-augment-funcs-3-phase-shift-2-lr1e5-l21e2-128batch-run2_2020_11_14_17_13_27_fit_log.csv',
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/CNN-ResBlock-3-syn-even-skip-augment_funcs-3-timestep_delta_4-phase_shift-2-lr1e5-l2e2-128batch-lr_Scheduler_10_epochs_factor_01_2020_11_10_09_25_32/CNN-ResBlock-3-syn-even-skip-augment_funcs-3-timestep_delta_4-phase_shift-2-lr1e5-l2e2-128batch-lr_Scheduler_10_epochs_factor_01_2020_11_10_09_25_32_fit_log.csv',
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/CNN-ResBlock-3-syn-even-augment-funcs-3-phase-shift-2-lr1e5-l21e2-128batch-run4_2020_11_15_19_02_16/CNN-ResBlock-3-syn-even-augment-funcs-3-phase-shift-2-lr1e5-l21e2-128batch-run4_2020_11_15_19_02_16_fit_log.csv',
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/CNN-ResBlock-3-syn-even-augment-funcs-3-phase-shift-2-lr1e5-l21e2-128batch-run5_2020_11_15_20_06_35/CNN-ResBlock-3-syn-even-augment-funcs-3-phase-shift-2-lr1e5-l21e2-128batch-run5_2020_11_15_20_06_35_fit_log.csv',

    # 2-layer, no L2
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_2-augment_funcs_3-lr1e5_2020_11_21_17_18_57/cnn-baseline-syn-even-phase_shift_2-augment_funcs_3-lr1e5_2020_11_21_17_18_57_fit_log.csv',
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_2-augment_funcs_3-lr1e5-128batch-run2_2020_11_22_10_57_27/cnn-baseline-syn-even-phase_shift_2-augment_funcs_3-lr1e5-128batch-run2_2020_11_22_10_57_27_fit_log.csv',
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_2-augment_funcs_3-lr1e5-128batch-run3_2020_11_22_10_58_28/cnn-baseline-syn-even-phase_shift_2-augment_funcs_3-lr1e5-128batch-run3_2020_11_22_10_58_28_fit_log.csv',

    # auc schedule
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_2-augment_funcs_1-lr1e5-l21e2-128batch-auc_scheduler-run2_2020_11_22_11_42_06/cnn-baseline-syn-even-phase_shift_2-augment_funcs_1-lr1e5-l21e2-128batch-auc_scheduler-run2_2020_11_22_11_42_06_fit_log.csv',

    # auc schedule favctor 0.5
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_2-augment_funcs_1-lr1e5-l21e2-128batch-auc_scheduler-factor_05_2020_11_22_12_36_03/cnn-baseline-syn-even-phase_shift_2-augment_funcs_1-lr1e5-l21e2-128batch-auc_scheduler-factor_05_2020_11_22_12_36_03_fit_log.csv',

    # BN
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_2-augment_funcs_3-lr1e5-l21e2-128batch-use_batch_norm_2020_11_22_13_00_24/cnn-baseline-syn-even-phase_shift_2-augment_funcs_3-lr1e5-l21e2-128batch-use_batch_norm_2020_11_22_13_00_24_fit_log.csv',
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_2-augment_funcs_3-lr1e5-l21e2-128batch-use_batch_norm-run2_2020_11_22_13_01_39/cnn-baseline-syn-even-phase_shift_2-augment_funcs_3-lr1e5-l21e2-128batch-use_batch_norm-run2_2020_11_22_13_01_39_fit_log.csv',
]
label_list = [
    '1 layer',
    '2 layer',
    # '4 layer',
    # '8 layer',
    'proposed Model',
    '2 layer no L2',
    'auc_schedule, thr=0.99, factor=0.1',
    'auc_schedule, thr=0.99, factor=0.5',
    'Batch norm'

]



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

def average(fit_hist_list,indexes_to_avg):
    fit_avg = [fit_hist for i,fit_hist in enumerate(fit_hist_list) if i in indexes_to_avg]

    num_avg = len(fit_avg)
    arr_length = min([len(hist['loss'])  for hist in fit_avg])
    fit_hist_avg = {}
    for key in fit_avg[0].keys():
        if key == 'lr':
            continue
        arr = np.zeros(arr_length)
        for i in range(num_avg):
            arr += np.array(fit_avg[i][key][:arr_length])
        fit_hist_avg[key] = (1.0 / num_avg) * arr
    return fit_hist_avg

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
        label_train = '{} train'.format(label_list[i])
        # label_train = '{}'.format(label_list[i])
        plt.plot(epochs, fit_dict['{}'.format(metric)][start_epoch:final_epoch], linestyle='-', lw=2,
                 label=label_train)
        color = plt.gca().lines[-1].get_color()
        # validation loss
        # label_valid = 'Model {} valid'.format(i + 1)
        label_valid = '{} validation'.format(label_list[i])
        plt.plot(epochs, fit_dict['val_{}'.format(metric)][start_epoch:final_epoch],linestyle='--', lw=2,
                 color=color)
        val_values = fit_dict['val_{}'.format(metric)][start_epoch:final_epoch]
        if i == 2 and metric == 'auc':
            plt.scatter(epochs[-1:],val_values[-1:], s=50 ,color=color, marker='^')
        # if i == 2 and metric == 'auc':
        #     epoch_lr_sched = [e if 0 == e % 10 for e in epochs]
        #     plt.scatter(epoch_lr_sched,val_values[-1:], s=50 ,color=color, marker='^')

    # plt.title(metric, fontsize=30)
    plt.grid()
    if start_epoch != 0:
        plt.xticks(np.arange(start_epoch, final_epoch, 25))
    plt.legend(loc="best",fontsize=8)
    plt.xlabel('Epochs', fontsize=15)
    metric = 'ROC-AUC' if metric == 'auc' else metric
    plt.ylabel('{}'.format(metric),fontsize=15)
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


dict1 = average(fit_history_list,[0,1,2,3])
dict2 = average(fit_history_list,[4,5,6,7])
dict3 = average(fit_history_list,[8,9,10,11,12])
dict4 = average(fit_history_list,[13,14,15])
dict5 = average(fit_history_list,[16])
dict6 = average(fit_history_list,[17])
dict7 = average(fit_history_list,[18,19])


# dict3 = average(fit_history_list,[8,9,10])
# dict4 = average(fit_history_list,[11,12,13])
# dict5 = average(fit_history_list,[14,15,16,17,18])
# dict6 = average(fit_history_list,[19,20,21])
# dict7 = average(fit_history_list,[22])

fit_history_list = [dict1, dict2, dict3, dict4, dict5, dict6, dict7]
plot_metric('accuracy', fit_history_list)
plot_metric('loss', fit_history_list, start_epoch=5)
plot_metric('auc', fit_history_list)

print('')

#
# DELTA_INTERVAL = 100
# _ = np.mean([x_aug - x for x_aug,x in zip(fit_history_list[0]['val_auc'][-20:],fit_history_list[1]['val_auc'][-20:])])
# print(_)
