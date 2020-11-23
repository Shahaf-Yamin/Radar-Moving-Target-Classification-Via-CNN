import csv
import collections
import matplotlib.pyplot as plt
import os
import numpy as np
import pathlib

# path_list = [
#     # flips
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/Sweep-cnn-baseline-syn-even-lr1e5-128batch-augment_funcs_list-flips_2020_11_19_20_10_15/Sweep-cnn-baseline-syn-even-lr1e5-128batch-augment_funcs_list-flips_2020_11_19_20_10_15_augment_funcs__fit_log.csv',
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/Sweep-cnn-baseline-syn-even-lr1e5-128batch-augment_funcs_list-flips_2020_11_19_20_10_15/Sweep-cnn-baseline-syn-even-lr1e5-128batch-augment_funcs_list-flips_2020_11_19_20_10_15_augment_funcs_\'flip_image\'_fit_log.csv',
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/Sweep-cnn-baseline-syn-even-lr1e5-128batch-augment_funcs_list-flips_2020_11_19_20_10_15/Sweep-cnn-baseline-syn-even-lr1e5-128batch-augment_funcs_list-flips_2020_11_19_20_10_15_augment_funcs_\'horiz_flip\'_fit_log.csv',
#     # timestep
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/Sweep-cnn-baseline-syn-even-lr1e5-128batch-timestep_delta-expansion_number_2020_11_19_18_51_25/Sweep-cnn-baseline-syn-even-lr1e5-128batch-timestep_delta-expansion_number_2020_11_19_18_51_25_augment_expansion_number_1_fit_log.csv',
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/Sweep-cnn-baseline-syn-even-lr1e5-128batch-timestep_delta-expansion_number_2020_11_19_18_51_25/Sweep-cnn-baseline-syn-even-lr1e5-128batch-timestep_delta-expansion_number_2020_11_19_18_51_25_augment_expansion_number_2_fit_log.csv',
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/Sweep-cnn-baseline-syn-even-lr1e5-128batch-timestep_delta-expansion_number_2020_11_19_18_51_25/Sweep-cnn-baseline-syn-even-lr1e5-128batch-timestep_delta-expansion_number_2020_11_19_18_51_25_augment_expansion_number_4_fit_log.csv',
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/Sweep-cnn-baseline-syn-even-lr1e5-128batch-timestep_delta-expansion_number_2020_11_19_18_51_25/Sweep-cnn-baseline-syn-even-lr1e5-128batch-timestep_delta-expansion_number_2020_11_19_18_51_25_augment_expansion_number_8_fit_log.csv',
#     # normal
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/Sweep-cnn-baseline-syn-even-lr1e5-128batch-normal-expansion_number_2020_11_19_18_48_36/Sweep-cnn-baseline-syn-even-lr1e5-128batch-normal-expansion_number_2020_11_19_18_48_36_augment_expansion_number_1_fit_log.csv',
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/Sweep-cnn-baseline-syn-even-lr1e5-128batch-normal-expansion_number_2020_11_19_18_48_36/Sweep-cnn-baseline-syn-even-lr1e5-128batch-normal-expansion_number_2020_11_19_18_48_36_augment_expansion_number_2_fit_log.csv',
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/Sweep-cnn-baseline-syn-even-lr1e5-128batch-normal-expansion_number_2020_11_19_18_48_36/Sweep-cnn-baseline-syn-even-lr1e5-128batch-normal-expansion_number_2020_11_19_18_48_36_augment_expansion_number_4_fit_log.csv',
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/Sweep-cnn-baseline-syn-even-lr1e5-128batch-normal-expansion_number_2020_11_19_18_48_36/Sweep-cnn-baseline-syn-even-lr1e5-128batch-normal-expansion_number_2020_11_19_18_48_36_augment_expansion_number_8_fit_log.csv',
#     # pca
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/Sweep-cnn-baseline-syn-even-lr1e5-128batch-pca-num_segments_2020_11_19_18_56_13/Sweep-cnn-baseline-syn-even-lr1e5-128batch-pca-num_segments_2020_11_19_18_56_13_num_of_pca_segments_1_fit_log.csv',
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/Sweep-cnn-baseline-syn-even-lr1e5-128batch-pca-num_segments_2020_11_19_18_56_13/Sweep-cnn-baseline-syn-even-lr1e5-128batch-pca-num_segments_2020_11_19_18_56_13_num_of_pca_segments_2_fit_log.csv',
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/Sweep-cnn-baseline-syn-even-lr1e5-128batch-pca-num_segments_2020_11_19_18_56_13/Sweep-cnn-baseline-syn-even-lr1e5-128batch-pca-num_segments_2020_11_19_18_56_13_num_of_pca_segments_4_fit_log.csv',
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/Sweep-cnn-baseline-syn-even-lr1e5-128batch-pca-num_segments_2020_11_19_18_56_13/Sweep-cnn-baseline-syn-even-lr1e5-128batch-pca-num_segments_2020_11_19_18_56_13_num_of_pca_segments_8_fit_log.csv',
#     # rfs
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/Sweep-cnn-baseline-syn-even-lr1e5-128batch-num_of_phase_time_shift_list_2020_11_19_18_45_05/Sweep-cnn-baseline-syn-even-lr1e5-128batch-num_of_phase_time_shift_list_2020_11_19_18_45_05_num_of_phase_time_shift_1_fit_log.csv',
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/Sweep-cnn-baseline-syn-even-lr1e5-128batch-num_of_phase_time_shift_list_2020_11_19_18_45_05/Sweep-cnn-baseline-syn-even-lr1e5-128batch-num_of_phase_time_shift_list_2020_11_19_18_45_05_num_of_phase_time_shift_2_fit_log.csv',
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/Sweep-cnn-baseline-syn-even-lr1e5-128batch-num_of_phase_time_shift_list_2020_11_19_18_45_05/Sweep-cnn-baseline-syn-even-lr1e5-128batch-num_of_phase_time_shift_list_2020_11_19_18_45_05_num_of_phase_time_shift_4_fit_log.csv',
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/Sweep-cnn-baseline-syn-even-lr1e5-128batch-num_of_phase_time_shift_list_2020_11_19_18_45_05/Sweep-cnn-baseline-syn-even-lr1e5-128batch-num_of_phase_time_shift_list_2020_11_19_18_45_05_num_of_phase_time_shift_8_fit_log.csv',
#     # regularization term
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/Sweep-cnn-baseline-syn-even-lr1e5-128batch-Reg_term_2020_11_19_20_14_59/Sweep-cnn-baseline-syn-even-lr1e5-128batch-Reg_term_2020_11_19_20_14_59_Regularization_term_0.1_fit_log.csv',
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/Sweep-cnn-baseline-syn-even-lr1e5-128batch-Reg_term_2020_11_19_20_14_59/Sweep-cnn-baseline-syn-even-lr1e5-128batch-Reg_term_2020_11_19_20_14_59_Regularization_term_0.01_fit_log.csv',
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/Sweep-cnn-baseline-syn-even-lr1e5-128batch-Reg_term_2020_11_19_20_14_59/Sweep-cnn-baseline-syn-even-lr1e5-128batch-Reg_term_2020_11_19_20_14_59_Regularization_term_0.001_fit_log.csv',
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/Sweep-cnn-baseline-syn-even-lr1e5-128batch-Reg_term_2020_11_19_20_14_59/Sweep-cnn-baseline-syn-even-lr1e5-128batch-Reg_term_2020_11_19_20_14_59_Regularization_term_0.0001_fit_log.csv',
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/Sweep-cnn-baseline-syn-even-lr1e5-128batch-Reg_term_2020_11_19_20_14_59/Sweep-cnn-baseline-syn-even-lr1e5-128batch-Reg_term_2020_11_19_20_14_59_Regularization_term_1e-06_fit_log.csv',
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/Sweep-cnn-baseline-syn-even-lr1e5-128batch-Reg_term_2020_11_19_20_14_59/Sweep-cnn-baseline-syn-even-lr1e5-128batch-Reg_term_2020_11_19_20_14_59_Regularization_term_1e-08_fit_log.csv',
# ]

# path_list = [
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_3-lr1e5-128batch_2020_11_20_17_36_22/cnn-baseline-syn-even-phase_shift_3-lr1e5-128batch_2020_11_20_17_36_22_fit_log.csv',
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_3-lr1e5-128batch_2020_11_20_17_37_24/cnn-baseline-syn-even-phase_shift_3-lr1e5-128batch_2020_11_20_17_37_24.csv',
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-timestep_shift_3-lr1e5-128batch_2020_11_20_17_38_57/cnn-baseline-syn-even-timestep_shift_3-lr1e5-128batch_2020_11_20_17_38_57_fit_log.csv',
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-pca_3-lr1e5-128batch_2020_11_20_17_40_53/cnn-baseline-syn-even-pca_3-lr1e5-128batch_2020_11_20_17_40_53_fit_log.csv',
# ]
#
# # Data Size, Same Model
# path_list = [
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-lr1e5-l21e3_2020_11_21_12_10_25/cnn-baseline-lr1e5-l21e3_2020_11_21_12_10_25_fit_log.csv',
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-lr1e5-l21e3_2020_11_21_12_12_13/cnn-baseline-syn-even-lr1e5-l21e3_2020_11_21_12_12_13_fit_log.csv',
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_2-lr1e5-l21e3_2020_11_21_12_14_42/cnn-baseline-syn-even-phase_shift_2-lr1e5-l21e3_2020_11_21_12_14_42_fit_log.csv',
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_2-vertical_flip-lr1e5-l21e3_2020_11_21_12_20_50/cnn-baseline-syn-even-phase_shift_2-vertical_flip-lr1e5-l21e3_2020_11_21_12_20_50_fit_log.csv',
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_2-vertical_flip-horiz_flip-lr1e5-l21e3_2020_11_21_12_55_38/cnn-baseline-syn-even-phase_shift_2-vertical_flip-horiz_flip-lr1e5-l21e3_2020_11_21_12_55_38_fit_log.csv',
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_2-augment_funcs_1-lr1e5-l21e3_2020_11_21_13_00_32/cnn-baseline-syn-even-phase_shift_2-augment_funcs_1-lr1e5-l21e3_2020_11_21_13_00_32_fit_log.csv',
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_2-augment_funcs_2-lr1e5-l21e3_2020_11_21_13_06_32/cnn-baseline-syn-even-phase_shift_2-augment_funcs_2-lr1e5-l21e3_2020_11_21_13_06_32_fit_log.csv',
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_2-augment_funcs_3-lr1e5-l21e3_2020_11_21_13_07_40/cnn-baseline-syn-even-phase_shift_2-augment_funcs_3-lr1e5-l21e3_2020_11_21_13_07_40_fit_log.csv',
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_3-augment_funcs_3-lr1e5-l21e3_2020_11_21_14_09_18/cnn-baseline-syn-even-phase_shift_3-augment_funcs_3-lr1e5-l21e3_2020_11_21_14_09_18_fit_log.csv',
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_4-augment_funcs_4-lr1e5-l21e3_2020_11_21_16_46_03/cnn-baseline-syn-even-phase_shift_4-augment_funcs_4-lr1e5-l21e3_2020_11_21_16_46_03_fit_log.csv',
#     # NO L2 400K
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_2-augment_funcs_3-lr1e5_2020_11_21_17_18_57/cnn-baseline-syn-even-phase_shift_2-augment_funcs_3-lr1e5_2020_11_21_17_18_57_fit_log.csv',
# ]

# path_list = [
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-1layers-syn-even-augment_funcs_3-phase_shift_2-lr1e5-l21e3-128batch_2020_11_20_19_24_31/cnn-baseline-1layers-syn-even-augment_funcs_3-phase_shift_2-lr1e5-l21e3-128batch_2020_11_20_19_24_31_fit_log.csv',
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-2layers-syn-even-augment_funcs_3-phase_shift_2-lr1e5-l21e3-128batch_2020_11_20_19_22_55/cnn-baseline-2layers-syn-even-augment_funcs_3-phase_shift_2-lr1e5-l21e3-128batch_2020_11_20_19_22_55_fit_log.csv',
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-4layers-syn-even-augment_funcs_3-phase_shift_2-lr1e5-l21e3-128batch_2020_11_20_19_34_32/cnn-baseline-4layers-syn-even-augment_funcs_3-phase_shift_2-lr1e5-l21e3-128batch_2020_11_20_19_34_32_fit_log.csv',
#     '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-8layers-syn-even-augment_funcs_3-phase_shift_2-lr1e5-l21e3-128batch_2020_11_20_23_56_33/cnn-baseline-8layers-syn-even-augment_funcs_3-phase_shift_2-lr1e5-l21e3-128batch_2020_11_20_23_56_33_fit_log.csv',
# ]

# layers - 400K
path_list = [
    # 1layer
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-1layers-syn-even-augment_funcs_3-phase_shift_2-lr1e5-l21e3-128batch_2020_11_20_19_24_31/cnn-baseline-1layers-syn-even-augment_funcs_3-phase_shift_2-lr1e5-l21e3-128batch_2020_11_20_19_24_31_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-1layer-syn-even-phase_shift_2-augment_funcs_3-lr1e5-l21e3-run2_2020_11_21_18_27_45/cnn-baseline-1layer-syn-even-phase_shift_2-augment_funcs_3-lr1e5-l21e3-run2_2020_11_21_18_27_45_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-1layer-syn-even-phase_shift_2-augment_funcs_3-lr1e5-l21e3-run3_2020_11_21_18_28_58/cnn-baseline-1layer-syn-even-phase_shift_2-augment_funcs_3-lr1e5-l21e3-run3_2020_11_21_18_28_58_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-1layer-syn-even-phase_shift_2-augment_funcs_3-lr1e5-l21e3-run4_2020_11_21_18_32_37/cnn-baseline-1layer-syn-even-phase_shift_2-augment_funcs_3-lr1e5-l21e3-run4_2020_11_21_18_32_37_fit_log.csv',

    # 2layer
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_2-augment_funcs_3-lr1e5-l21e3_2020_11_21_13_07_40/cnn-baseline-syn-even-phase_shift_2-augment_funcs_3-lr1e5-l21e3_2020_11_21_13_07_40_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-2layers-syn-even-augment_funcs_3-phase_shift_2-lr1e5-l21e3-128batch_2020_11_20_19_22_55/cnn-baseline-2layers-syn-even-augment_funcs_3-phase_shift_2-lr1e5-l21e3-128batch_2020_11_20_19_22_55_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-2layer-syn-even-phase_shift_2-augment_funcs_3-lr1e5-l21e3-run3_2020_11_21_18_38_22/cnn-baseline-2layer-syn-even-phase_shift_2-augment_funcs_3-lr1e5-l21e3-run3_2020_11_21_18_38_22_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-2layer-syn-even-phase_shift_2-augment_funcs_3-lr1e5-l21e3-run4_2020_11_21_18_42_11/cnn-baseline-2layer-syn-even-phase_shift_2-augment_funcs_3-lr1e5-l21e3-run4_2020_11_21_18_42_11_fit_log.csv',

    # 4layer
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-4layers-syn-even-augment_funcs_3-phase_shift_2-lr1e5-l21e3-128batch_2020_11_20_19_34_32/cnn-baseline-4layers-syn-even-augment_funcs_3-phase_shift_2-lr1e5-l21e3-128batch_2020_11_20_19_34_32_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-4layer-syn-even-phase_shift_2-augment_funcs_3-lr1e5-l21e3-run2_2020_11_21_18_46_23/cnn-baseline-4layer-syn-even-phase_shift_2-augment_funcs_3-lr1e5-l21e3-run2_2020_11_21_18_46_23_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-4layers-syn-even-phase_shift_2-augment_funcs_3-lr1e5-l21e3-run3_2020_11_21_20_29_19/cnn-baseline-4layers-syn-even-phase_shift_2-augment_funcs_3-lr1e5-l21e3-run3_2020_11_21_20_29_19_fit_log.csv',
    #
    # # 8layer
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-8layers-syn-even-augment_funcs_3-phase_shift_2-lr1e5-l21e3-128batch_2020_11_20_23_56_33/cnn-baseline-8layers-syn-even-augment_funcs_3-phase_shift_2-lr1e5-l21e3-128batch_2020_11_20_23_56_33_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-8layers-syn-even-augment_funcs_3-phase_shift_2-lr1e5-l21e3-128batch-run2_2020_11_21_23_43_32/cnn-baseline-8layers-syn-even-augment_funcs_3-phase_shift_2-lr1e5-l21e3-128batch-run2_2020_11_21_23_43_32_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-8layers-syn-even-augment_funcs_3-phase_shift_2-lr1e5-l21e3-128batch-run3_2020_11_21_23_44_55/cnn-baseline-8layers-syn-even-augment_funcs_3-phase_shift_2-lr1e5-l21e3-128batch-run3_2020_11_21_23_44_55_fit_log.csv',



    # no L2
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_2-augment_funcs_3-lr1e5_2020_11_21_17_18_57/cnn-baseline-syn-even-phase_shift_2-augment_funcs_3-lr1e5_2020_11_21_17_18_57_fit_log.csv',

    # Proposed Model
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/CNN-ResBlock-3-syn-even-skip-augment_funcs-3-phase_shift-2-lre4-l2e2-128batch-lr_scheduler_mod10_2020_11_07_23_30_25/CNN-ResBlock-3-syn-even-skip-augment_funcs-3-phase_shift-2-lre4-l2e2-128batch-lr_scheduler_mod10_2020_11_07_23_30_25_fit_log.csv',
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/CNN-ResBlock-3-syn-even-augment-funcs-3-phase-shift-2-lr1e5-l21e2-128batch-run2_2020_11_14_17_13_27/CNN-ResBlock-3-syn-even-augment-funcs-3-phase-shift-2-lr1e5-l21e2-128batch-run2_2020_11_14_17_13_27_fit_log.csv',
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/CNN-ResBlock-3-syn-even-skip-augment_funcs-3-timestep_delta_4-phase_shift-2-lr1e5-l2e2-128batch-lr_Scheduler_10_epochs_factor_01_2020_11_10_09_25_32/CNN-ResBlock-3-syn-even-skip-augment_funcs-3-timestep_delta_4-phase_shift-2-lr1e5-l2e2-128batch-lr_Scheduler_10_epochs_factor_01_2020_11_10_09_25_32_fit_log.csv',
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/CNN-ResBlock-3-syn-even-augment-funcs-3-phase-shift-2-lr1e5-l21e2-128batch-run4_2020_11_15_19_02_16/CNN-ResBlock-3-syn-even-augment-funcs-3-phase-shift-2-lr1e5-l21e2-128batch-run4_2020_11_15_19_02_16_fit_log.csv',
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/CNN-ResBlock-3-syn-even-augment-funcs-3-phase-shift-2-lr1e5-l21e2-128batch-run5_2020_11_15_20_06_35/CNN-ResBlock-3-syn-even-augment-funcs-3-phase-shift-2-lr1e5-l21e2-128batch-run5_2020_11_15_20_06_35_fit_log.csv',
]

# data size - 2layers
path_list = [
    #90K
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_2-vertical_flip-lr1e5-l21e3_2020_11_21_12_20_50/cnn-baseline-syn-even-phase_shift_2-vertical_flip-lr1e5-l21e3_2020_11_21_12_20_50_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_2-vertical-lr1e5-l21e3-run2_2020_11_21_22_31_48/cnn-baseline-syn-even-phase_shift_2-vertical-lr1e5-l21e3-run2_2020_11_21_22_31_48_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_2-vertical-lr1e5-l21e3-run3_2020_11_21_22_34_38/cnn-baseline-syn-even-phase_shift_2-vertical-lr1e5-l21e3-run3_2020_11_21_22_34_38_fit_log.csv',
    #135K
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_2-vertical_flip-horiz_flip-lr1e5-l21e3_2020_11_21_12_55_38/cnn-baseline-syn-even-phase_shift_2-vertical_flip-horiz_flip-lr1e5-l21e3_2020_11_21_12_55_38_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_2-vertical_horizontal-lr1e5-l21e3-run2_2020_11_21_19_18_00/cnn-baseline-syn-even-phase_shift_2-vertical_horizontal-lr1e5-l21e3-run2_2020_11_21_19_18_00_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_2-vertical_horizontal-lr1e5-l21e3-run3_2020_11_21_19_19_01/cnn-baseline-syn-even-phase_shift_2-vertical_horizontal-lr1e5-l21e3-run3_2020_11_21_19_19_01_fit_log.csv',
    #225K
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_2-augment_funcs_1-lr1e5-l21e3_2020_11_21_13_00_32/cnn-baseline-syn-even-phase_shift_2-augment_funcs_1-lr1e5-l21e3_2020_11_21_13_00_32_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_2-augment_funcs_1-lr1e5-l21e3-run2_2020_11_21_20_19_37/cnn-baseline-syn-even-phase_shift_2-augment_funcs_1-lr1e5-l21e3-run2_2020_11_21_20_19_37_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_2-augment_funcs_1-lr1e5-l21e3-run3_2020_11_21_20_20_24/cnn-baseline-syn-even-phase_shift_2-augment_funcs_1-lr1e5-l21e3-run3_2020_11_21_20_20_24_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_2-augment_funcs_1-lr1e5-l21e3-run4_2020_11_21_20_21_24/cnn-baseline-syn-even-phase_shift_2-augment_funcs_1-lr1e5-l21e3-run4_2020_11_21_20_21_24_fit_log.csv'
    # 320K
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_2-augment_funcs_2-lr1e5-l21e3_2020_11_21_13_06_32/cnn-baseline-syn-even-phase_shift_2-augment_funcs_2-lr1e5-l21e3_2020_11_21_13_06_32_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_2-augment_funcs_2-lr1e5-l21e3-run2_2020_11_21_20_22_43/cnn-baseline-syn-even-phase_shift_2-augment_funcs_2-lr1e5-l21e3-run2_2020_11_21_20_22_43_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_2-augment_funcs_2-lr1e5-l21e3-run3_2020_11_21_20_26_21/cnn-baseline-syn-even-phase_shift_2-augment_funcs_2-lr1e5-l21e3-run3_2020_11_21_20_26_21_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-2layers-syn-even-phase_shift_2-augment_funcs_2-lr1e5-l21e3-run4_2020_11_21_21_04_34/cnn-baseline-2layers-syn-even-phase_shift_2-augment_funcs_2-lr1e5-l21e3-run4_2020_11_21_21_04_34_fit_log.csv',
    # 540K
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_3-augment_funcs_3-lr1e5-l21e3_2020_11_21_14_09_18/cnn-baseline-syn-even-phase_shift_3-augment_funcs_3-lr1e5-l21e3_2020_11_21_14_09_18_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_3-augment_funcs_3-lr1e5-l21e3-run2_2020_11_21_21_19_10/cnn-baseline-syn-even-phase_shift_3-augment_funcs_3-lr1e5-l21e3-run2_2020_11_21_21_19_10_fit_log.csv',
    # '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_3-augment_funcs_3-lr1e5-l21e3-run3_2020_11_21_21_24_06/cnn-baseline-syn-even-phase_shift_3-augment_funcs_3-lr1e5-l21e3-run3_2020_11_21_21_24_06_fit_log.csv',
    # 815K
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-phase_shift_4-augment_funcs_4-lr1e5-l21e3_2020_11_21_16_46_03/cnn-baseline-syn-even-phase_shift_4-augment_funcs_4-lr1e5-l21e3_2020_11_21_16_46_03_fit_log.csv',
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-augment_funcs_4-phase_shift_4-lr1e5-l21e3-128batch-run2_2020_11_22_10_14_13/cnn-baseline-syn-even-augment_funcs_4-phase_shift_4-lr1e5-l21e3-128batch-run2_2020_11_22_10_14_13_fit_log.csv',
    '/Users/stefanfeintuch/Desktop/Research/Radar/graphs/cnn-baseline-syn-even-augment_funcs_4-phase_shift_4-lr1e5-l21e3-128batch-run3_2020_11_22_10_14_43/cnn-baseline-syn-even-augment_funcs_4-phase_shift_4-lr1e5-l21e3-128batch-run3_2020_11_22_10_14_43_fit_log.csv',
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


result_file_list = []
result_file_name_list = []
for path in path_list:
    with open(path, 'r') as csv_file:
        p = pathlib.Path(path)
        result_file_name_list.append(p.name)
        result_csv = csv.reader(csv_file, delimiter=',', quotechar='|')
        csv_dict = csv_to_dict(result_csv)
        for key in csv_dict.keys():
            if key == 'epoch':
                csv_dict[key] = [int(x) for x in csv_dict[key]]
            else:
                csv_dict[key] = [float(x) for x in csv_dict[key]]
        result_file_list.append(csv_dict)


# Mean val_AUC
# val_auc_list = []
# for name, res_csv in zip(result_file_name_list, result_file_list):
#     val_auc_list.append(np.array(res_csv['val_auc']))
# WINDOW_SIZE = 5
# val_auc_mean = np.mean(np.array(val_auc_list),0)
# peak_index = np.argmax(val_auc_mean)
# mean_max_auc = np.mean(val_auc_mean[peak_index - WINDOW_SIZE:peak_index + WINDOW_SIZE + 1])
# print('AUC: {}, peak index: {}'.format(mean_max_auc, peak_index))



print(70 * '#')
print('val_auc results:')
WINDOW_SIZE = 5
mean_max_auc_list = []
peak_index_list = []
for name, res_csv in zip(result_file_name_list, result_file_list):
    val_auc = np.array(res_csv['val_auc'])
    peak_index = np.argmax(val_auc)
    left = max(peak_index - WINDOW_SIZE, 0)
    right = min(peak_index + WINDOW_SIZE + 1,len(val_auc))
    mean_max_auc = np.mean(val_auc[left:right])
    print('{}: {}, peak index: {}'.format(name, mean_max_auc, peak_index))
    mean_max_auc_list.append(mean_max_auc)
    peak_index_list.append(peak_index)
print('MEAN AUC: {}, peak: {}'.format(np.mean(mean_max_auc_list), np.mean(peak_index_list)))
