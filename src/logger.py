from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime


def get_logger():
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.INFO)

    os.makedirs("logs", exist_ok=True)
    fh = logging.FileHandler(f"logs/{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log")
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


logger = get_logger()

def calculate_metrics(y_true, y_pred, duration):
    res = pd.DataFrame(data=np.zeros((1, 4), dtype=float), index=[0],
                       columns=['precision', 'recall', 'duration', 'f1'])
    res['duration'] = duration
    res['precision'] = precision_score(y_true, y_pred, average='binary')
    res['recall'] = recall_score(y_true, y_pred, average='binary')
    res['f1'] = f1_score(y_true, y_pred, average='binary')

    print(res)
    return res

def plot_predictions(y_pred, y_true, filename):
    fig, ax = plt.subplots()
    t = list(range(len(y_pred)))
    ax.plot(t, y_true, "b-", t, y_pred, "r.")
    fig.savefig(filename)
    plt.close(fig)


def save_predictions(y_true, y_pred, filename):
    with open(filename, "w+") as file:
        for line in [y_true, y_pred]:
            for elem in line:
                file.write(f"{elem} ")
            file.write("\n")


def plot_epochs_metric(hist, file_name, metric='loss'):
    fig, ax = plt.subplots()
    ax.plot(hist.history[metric])
    ax.plot(hist.history['val_' + metric])
    ax.set_title('model ' + metric)
    ax.set_ylabel(metric, fontsize='large')
    ax.set_xlabel('epoch', fontsize='large')
    ax.legend(['train', 'val'], loc='upper left')
    fig.savefig(file_name, bbox_inches='tight')
    plt.close(fig)

def log_predicions(output_directory, y_pred, y_true, duration, fold, round):
    df_metrics = calculate_metrics(y_true, y_pred, duration)
    df_metrics.to_csv(os.path.join(output_directory, f'{fold}-{round}_df_metrics.csv'), index=False)

    plot_predictions(y_pred, y_true, os.path.join(output_directory, f'{fold}-{round}predictions.png'))
    save_predictions(y_true, y_pred, os.path.join(output_directory,f"{fold}-{round}predictions.txt"))

    return df_metrics

def save_logs(output_directory, hist, y_pred, y_true, duration, fold):
    hist_df = pd.DataFrame(hist.history)

    index_best_model = hist_df['val_loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 4), dtype=float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss',
                                         'best_model_learning_rate', 'best_model_nb_epoch'])

    loss = row_best_model['val_loss']

    df_metrics = calculate_metrics(y_true, y_pred, duration)
    df_metrics['loss'] = loss

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = loss
    df_best_model['best_model_nb_epoch'] = index_best_model
    df_best_model['best_model_learning_rate'] = row_best_model['learning_rate']

    hist_df.to_csv(os.path.join(output_directory, f'{fold}_history.csv'), index=False)
    df_metrics.to_csv(output_directory + f'{fold}_df_metrics.csv', index=False)
    df_best_model.to_csv(os.path.join(output_directory, f'{fold}df_best_model.csv'), index=False)

    # plot losses
    plot_epochs_metric(hist, os.path.join(output_directory, f'{fold}epochs_loss.png'))
    plot_predictions(y_pred, y_true, os.path.join(output_directory, f'{fold}predictions.png'))
    save_predictions(y_true, y_pred, os.path.join(output_directory,f"{fold}predictions.txt"))

    return df_metrics, loss
