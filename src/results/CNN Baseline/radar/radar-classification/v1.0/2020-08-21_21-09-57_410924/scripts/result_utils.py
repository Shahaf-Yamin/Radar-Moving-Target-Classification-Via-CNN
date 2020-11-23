from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import os
from tensorflow.keras.models import Model
import tensorflow_datasets as tfds

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
  plt.scatter(x, y, c=colors, alpha=.65,label=plot_labels)
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
  plt.plot(range(len(history_dict.history['val_auc'])), history_dict.history['val_auc'], linewidth=2,
           label='{}'.format('validation AUC'))
  plt.plot(range(len(history_dict.history['auc'])), history_dict.history['auc'], linewidth=2,
           label='{}'.format('train AUC'))

  plt.title('AUC ROC', fontsize=30)
  plt.legend(loc="best")
  plt.xlabel('Epochs', fontsize=20)
  plt.ylabel('AUC', fontsize=30)
  fig_path = os.path.join(path, 'AUC')
  plt.savefig(fig_path)

def stats(pred, actual,path):
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
  plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--', label = 'Random guess')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate', fontsize=18)
  plt.ylabel('True Positive Rate', fontsize=18)
  plt.title('Training set vs. Validation set ROC curves')
  plt.legend(loc="lower right", prop = {'size': 20})
  fig_path = os.path.join(path, 'AUC_statistics_analysis')
  plt.savefig(fig_path)

def make_predication(y):
  preds = np.empty(y.shape)

  for idx, val in enumerate(y):
    if val[0] < 0.5:
      preds[idx][0] = 0
    else:
      preds[idx][0] = 1

  return preds

def anaylse_accuracy_seprately(y,labels):

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

  print("Animal Accuracy= ", p_animal * 1.0 / animal)
  print("Human Accuracy= ", p_human * 1.0 / human)
  return preds

def analyse_model_performance(model, data, path, history,config):

  if not os.path.isdir(path):
    os.mkdir(path)

  print_graph(history, path)

  train_dataset = tfds.as_numpy(data['train'])
  val_dataset = tfds.as_numpy(data['train_eval'])

  '''
  Extract the model AUC performance
  '''

  predication_train = []
  train_input = []
  train_y = []
  predication_val   = []
  val_y = []
  val_input = []

  for index, batch in enumerate(train_dataset):
    train_input.append(batch[0])
    predication_train.append(model['train'].predict(batch[0]))
    train_y.append(batch[1])

  train_input = np.concatenate(train_input, axis=0)
  predication_train = np.concatenate(predication_train,axis=0)
  train_y = np.concatenate(train_y, axis=0)

  for index, batch in enumerate(val_dataset):
    val_input.append(batch[0])
    predication_val.append(model['train'].predict(batch[0]))
    val_y.append(batch[1])

  val_input = np.concatenate(val_input,axis=0)
  predication_val = np.concatenate(predication_val,axis=0)
  val_y = np.concatenate(val_y, axis=0)

  pred = [predication_train, predication_val]
  actual = [train_y, val_y]
  stats(pred, actual, path)


  '''
  Analyse Accuracy over both animal and human train
  '''
  print('Analysis of train accuracy:')
  anaylse_accuracy_seprately(pred[0], train_y)
  print('Analysis of validation accuracy:')
  anaylse_accuracy_seprately(pred[1], val_y)

  '''
  Visualize the high dimensions
  '''
  if config.with_TNSE:
    make_tsne(train_input, model, train_y,predication_train, path,exp_type='train', test=None)
    make_tsne(val_input, model, val_y,predication_val, path, exp_type='validation', test=None)



