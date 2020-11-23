from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import os
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json
import tensorflow_datasets as tfds
from datetime import datetime
from collections import OrderedDict
import csv


def make_tsne(data, model, labels, preds, path, exp_type, test=None):
    layer_name = 'dense_1'
    intermediate_layer_model = Model(inputs=model['train'].input,
                                     outputs=model['train'].get_layer(layer_name).output)

    intermediate_output = intermediate_layer_model.predict(data)

    tsne_data = intermediate_output
    preds = make_predication(preds)
    # possibly append the test data
    if test is not None and test.shape[0] != 0:
        test_output = intermediate_layer_model.predict(test)
        _td = np.concatenate((tsne_data, test_output))
        tsne_data = _td

    # assign a color for each type of signal
    colors = []
    plot_labels = []
    missc = 0

    if labels.shape[0] == 0:
        for i in range(data.shape[0]):
            colors.append('magenta')
            plot_labels.append('True Value')

    else:

        for i in range(labels.shape[0]):
            color = 'green' if labels[i] == 0 else 'blue'
            current_label = 'Animal' if labels[i] == 0 else 'Human'

            if labels[i] != preds[i]:
                color = 'red'  # 2
                current_label = 'False Predication'
                missc += 1

            colors.append(color)
            plot_labels.append(current_label)

    if test is not None and test.shape[0] != 0:
        for i in range(test.shape[0]):
            colors.append('cyan')
            plot_labels.append('Test')

    print("misclassify count=", missc)

    tmodel = TSNE(metric='cosine', perplexity=5, n_iter=1000)
    transformed = tmodel.fit_transform(tsne_data)

    # plot results

    figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])
    x = transformed[:, 0]
    y = transformed[:, 1]
    plt.scatter(x, y, c=colors, alpha=.65, label=plot_labels)
    fig_path = os.path.join(path, 'TNSE_statistics_' + exp_type)
    plt.savefig(fig_path)

def print_graph(history_dict, path):
    '''
    Plotting the accuracy over different sweeps back to back
    '''
    # Validation Printing
    plt.figure()
    plt.plot(range(len(history_dict.history['val_accuracy'])), history_dict.history['val_accuracy'], linewidth=2,
             label='{0}'.format('validation accuracy'))
    plt.plot(range(len(history_dict.history['accuracy'])), history_dict.history['accuracy'], linewidth=2,
             label='{0}'.format('train accuracy'))
    plt.title('Accuracy', fontsize=30)
    plt.legend(loc="best")
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Accuracy', fontsize=30)
    fig_path = os.path.join(path, 'Accuracy')
    plt.savefig(fig_path)

    plt.figure()
    plt.plot(range(len(history_dict.history['val_loss'])), history_dict.history['val_loss'], linewidth=2,
             label='{}'.format('validation loss'))
    plt.plot(range(len(history_dict.history['loss'])), history_dict.history['loss'], linewidth=2,
             label='{}'.format('train loss'))
    plt.title('Loss', fontsize=30)
    plt.legend(loc="best")
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Loss', fontsize=30)
    fig_path = os.path.join(path, 'Loss')
    plt.savefig(fig_path)

    plt.figure()
    train_auc = [(key, value) for key, value in history_dict.history.items() if 'auc' in key and 'val' not in key]
    val_auc = [(key, value) for key, value in history_dict.history.items() if 'val_auc' in key]
    plt.plot(range(len(val_auc[0][1])), val_auc[0][1], linewidth=2,
             label='{}'.format('validation AUC'))
    plt.plot(range(len(train_auc[0][1])), train_auc[0][1], linewidth=2,
             label='{}'.format('train AUC'))

    plt.title('AUC ROC', fontsize=30)
    plt.legend(loc="best")
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('AUC', fontsize=30)
    fig_path = os.path.join(path, 'AUC')
    plt.savefig(fig_path)

def stats(pred, actual, path):
    """
    Computes the model ROC-AUC score and plots the ROC curve.

    Arguments:
      pred -- {ndarray} -- model's probability predictions
      actual -- the true lables

    Returns:
      ROC curve graph and ROC-AUC score
    """
    plt.figure(figsize=(20, 10))
    fpr1, tpr1, _ = roc_curve(actual[0], pred[0])
    fpr2, tpr2, _ = roc_curve(actual[1], pred[1])
    roc_auc = [auc(fpr1, tpr1), auc(fpr2, tpr2)]
    lw = 2
    plt.plot(fpr1, tpr1, lw=lw, label='Training set (ROC-AUC = %0.2f)' % roc_auc[0])
    plt.plot(fpr2, tpr2, lw=lw, label='Validation set (ROC-AUC = %0.2f)' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--', label='Random guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title('Training set vs. Validation set ROC curves')
    plt.legend(loc="lower right", prop={'size': 20})
    fig_path = os.path.join(path, 'AUC_statistics_analysis')
    plt.savefig(fig_path)

def print_roc_auc_by_parameter(predictions_dict, param_name, title, graph_path):
    """
    Computes the model ROC-AUC score and plots the ROC curve.
    USED FOR PARAMETRIC SWEEP
    predictions_dict['param_name']['param']['train'/'valid']['y_pred'/'y_true']
    """
    plt.figure()
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', label='Random guess')
    for param_val in predictions_dict[param_name].keys():
        fpr_train, tpr_train, _ = roc_curve(predictions_dict[param_name][param_val]['train']['y_true'],
                                            predictions_dict[param_name][param_val]['train']['y_pred'])
        fpr_valid, tpr_valid, _ = roc_curve(predictions_dict[param_name][param_val]['valid']['y_true'],
                                            predictions_dict[param_name][param_val]['valid']['y_pred'])
        roc_auc = [auc(fpr_train, tpr_train), auc(fpr_valid, tpr_valid)]
        # train
        train_label = '{}={} Train (ROC-AUC = %0.2f)'.format(param_name, param_val) % roc_auc[0]
        train_label = (train_label[:65] + '..') if len(train_label) > 65 else train_label
        plt.plot(fpr_train, tpr_train, linestyle='-', lw=2,
                 label=train_label)
        color = plt.gca().lines[-1].get_color()
        # validation
        val_label = '{}={} Valid (ROC-AUC = %0.2f)'.format(param_name, param_val) % roc_auc[1]
        val_label = (val_label[:65] + '..') if len(val_label) > 65 else val_label
        plt.plot(fpr_valid, tpr_valid, linestyle='--', lw=2,
                 label=val_label, color=color)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title(plt.title(title, fontsize=30))
    plt.legend(loc="lower right")
    fig_path = os.path.join(graph_path, title)
    plt.savefig(fig_path)

def make_predication(y):
    preds = np.empty(y.shape)

    for idx, val in enumerate(y):
        if val[0] < 0.5:
            preds[idx][0] = 0
        else:
            preds[idx][0] = 1

    return preds

def anaylse_accuracy_seprately(y, labels):
    preds = make_predication(y)
    animal = 0
    human = 0
    p_animal = 0
    p_human = 0
    for label, predication in zip(labels, preds):
        if label == 0:
            animal = animal + 1
            if predication == 0:
                p_animal = p_animal + 1
        else:
            human = human + 1
            if predication == 1:
                p_human = p_human + 1

    try:
        animal_acc = p_animal * 1.0 / animal
    except:
        print("Zero Division!!, number of animals is zero")
        animal_acc = 0


    try:
        human_acc = p_human * 1.0 / human
    except:
        print("Zero Division!!, number of humans is zero")
        human_acc = 0

    print("Animal Accuracy= ", animal_acc)
    print("Human Accuracy= ", human_acc)
    return preds,animal_acc,human_acc

def analyse_model_performance(model, data, history, config, graph_path, res_dir):
    # graph_path path is the folder of all relevant graphs
    # path is the path for the specific (param_name, param) graphs
    path = os.path.join(graph_path, res_dir)
    if os.path.exists(path) is False:
        os.makedirs(path)

    print_graph(history, path)

    train_dataset = tfds.as_numpy(data['train'])
    val_dataset = tfds.as_numpy(data['train_eval'])

    '''
    Extract the model AUC performance
    '''

    predication_train = []
    train_input = []
    train_y = []
    predication_val = []
    val_y = []
    val_input = []

    for index, batch in enumerate(train_dataset):
        train_input.append(batch[0])
        predication_train.append(model['train'].predict(batch[0]))
        train_y.append(batch[1])

    train_input = np.concatenate(train_input, axis=0)
    predication_train = np.concatenate(predication_train, axis=0)
    train_y = np.concatenate(train_y, axis=0)

    for index, batch in enumerate(val_dataset):
        val_input.append(batch[0])
        predication_val.append(model['train'].predict(batch[0]))
        val_y.append(batch[1])

    val_input = np.concatenate(val_input, axis=0)
    predication_val = np.concatenate(predication_val, axis=0)
    val_y = np.concatenate(val_y, axis=0)

    pred = [predication_train, predication_val]
    actual = [train_y, val_y]
    stats(pred, actual, path)

    '''
    Analyse Accuracy over both animal and human train
    '''
    print('Analysis of train accuracy:')
    train_preds,train_animal_acc,train_human_acc = anaylse_accuracy_seprately(pred[0], train_y)
    print('Analysis of validation accuracy:')
    val_preds,val_animal_acc,val_human_acc = anaylse_accuracy_seprately(pred[1], val_y)

    '''
    Save results
    '''
    train_auc = np.max([value for key, value in history.history.items() if 'auc' in key and 'val' not in key])
    val_auc = np.max([value for key, value in history.history.items() if 'val_auc' in key])
    res_data = OrderedDict()
    res_data['Index'] = 0
    res_data['Train human accuracy'] = train_human_acc
    res_data['Train animal accuracy'] = train_animal_acc
    res_data['Train AUC'] = train_auc
    res_data['Validation human accuracy'] = val_human_acc
    res_data['Validation animal accuracy'] = val_animal_acc
    res_data['Validation AUC'] = val_auc

    '''
    Visualize the high dimensions
    '''
    if config.with_TNSE:
        make_tsne(train_input, model, train_y, predication_train, path, exp_type='train', test=None)
        make_tsne(val_input, model, val_y, predication_val, path, exp_type='validation', test=None)

    return res_data

def save_model(name,model):
    # save entire model
    model.save(name)
    print("Saved ENTIRE model to: {}".format(name))
    # # serialize model to JSON
    # model_json = model.to_json()
    # with open("{}.json".format(name), "w") as json_file:
    #     json_file.write(model_json)
    # # serialize weights to HDF5
    # model.save_weights("{}.h5".format(name))
    # print("Saved model to disk")

def save_best_model(snr_type,model):
    save_model(name='best_model_snr_{}'.format(snr_type),model=model)
    print("Model Improved results!!")

def load_best_model(snr_type,BEST_RESULT_DIR):
    # load json and create model
    json_file = open('{0}/best_model_snr_{1}.json.'.format(BEST_RESULT_DIR,snr_type), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights('{0}/best_model_snr_{1}.h5.'.format(BEST_RESULT_DIR,snr_type))
    return loaded_model

def compare_to_best_model_performance(result_data,model,BEST_RESULT_DIR,config):

    os.chdir(BEST_RESULT_DIR)

    if not os.path.exists('Best_performance_history.csv'):
        # Create csv with this file
        with open('Best_performance_history.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(result_data.keys())
            writer.writerow(result_data.values())
            save_best_model(snr_type=result_data['Snr type'],model=model['train'])
    else:
        # Update the csv with the current run results ( if needed )
        with open('Best_performance_history.csv') as csv_file:
            reader = csv.reader(csv_file)
            row_index = 0
            results = OrderedDict()
            for rows in reader:
                if row_index == 0:
                    for key in rows:
                        results[key] = []
                else:
                    for item, key in zip(rows, results.keys()):
                        results[key].append(item)
                row_index += 1
            '''
            Find best results for the current SNR experiment
            '''
            SNR_record_exist = False
            update_flag = False

            for index in range(len(results['Snr type']),0,-1):
                if results['Snr type'][index-1] == result_data['Snr type']:
                    Best_SNR_index = index-1
                    SNR_record_exist = True
                    break

            if not SNR_record_exist:
                # this case there isn't a previous SNR experiment
                update_flag = True
            elif results['Snr type'][Best_SNR_index] == result_data['Snr type'] and float(result_data['Validation AUC']) >= float(results['Validation AUC'][Best_SNR_index]):
                update_flag = True
                result_data['Index'] = len(results['Snr type'])

        if update_flag:
            with open('Best_performance_history.csv', 'a+', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(result_data.values())
            save_best_model(result_data['Snr type'], model['train'])

def print_sweep_by_parameter(hist_dict, param_name, metric_list, graph_path, title, ylabel=None):
    if ylabel is None:
        ylabel = title
    train_metric_dict = OrderedDict()
    val_metric_dict = OrderedDict()
    for metric in metric_list:
        train_metric_dict[metric] = {}
        val_metric_dict[metric] = {}
        for param_val in hist_dict[param_name].keys():
            if 'val' in metric:
                if 'val_auc' == metric:
                    val_auc = [(key, value) for key, value in hist_dict[param_name][param_val].items() if
                               'val_auc' in key]
                    val_metric_dict['val_auc'][param_val] = val_auc[0][1]
                else:
                    val_metric_dict[metric][param_val] = hist_dict[param_name][param_val][metric]
            else:
                if 'auc' == metric:
                    train_auc = [(key, value) for key, value in hist_dict[param_name][param_val].items() if
                                 'auc' in key and 'val' not in key]
                    train_metric_dict['auc'][param_val] = train_auc[0][1]
                else:
                    train_metric_dict[metric][param_val] = hist_dict[param_name][param_val][metric]
    metric_list_without_val = [metric for metric in metric_list if 'val' not in metric]
    plt.figure()
    for metric in metric_list_without_val:
        for param_val in train_metric_dict[metric].keys():
            # train
            train_label = '{}, {} = {}'.format(metric, param_name, param_val)
            train_label = (train_label[:65] + '..') if len(train_label) > 65 else train_label
            plt.plot(range(len(train_metric_dict[metric][param_val])), train_metric_dict[metric][param_val],
                     linestyle='-',
                     lw=2, label=train_label)
            color = plt.gca().lines[-1].get_color()
            # validation
            val_metric = 'val_' + metric
            val_label = '{}, {} = {}'.format(val_metric, param_name, param_val)
            val_label = (val_label[:65] + '..') if len(val_label) > 65 else val_label
            plt.plot(range(len(val_metric_dict[val_metric][param_val])), val_metric_dict[val_metric][param_val],
                     linestyle='--', lw=2, label=val_label, color=color)
    plt.grid(True)
    plt.title(title, fontsize=30)
    plt.legend(loc="best")
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel(ylabel, fontsize=30)
    fig_path = os.path.join(graph_path, title)
    plt.savefig(fig_path)

def get_predictions_dict_per_model(model, data):
    def get_predictions_and_labels_from_model(model, tf_dataset):
        dataset = tfds.as_numpy(tf_dataset)
        y_pred = []
        y_true = []
        for index, batch in enumerate(dataset):
            y_pred.append(model.predict(batch[0]))
            y_true.append(batch[1])

        y_pred_np = np.concatenate(y_pred, axis=0)
        y_true_np = np.concatenate(y_true, axis=0)

        return y_pred_np, y_true_np

    predictions_dict = OrderedDict()
    predictions_dict['train'] = {}
    predictions_dict['train']['y_pred'], predictions_dict['train']['y_true'] = get_predictions_and_labels_from_model(
        model, data['train'])
    predictions_dict['valid'] = {}
    predictions_dict['valid']['y_pred'], predictions_dict['valid']['y_true'] = get_predictions_and_labels_from_model(
        model, data['train_eval'])

    return predictions_dict
