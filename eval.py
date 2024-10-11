import os
import pandas as pd

survey_path = "{0}/survey_gamification".format(os.getcwd())
results = f'{os.getcwd()}/results_30'
fee_path = f'{results}/FEATURE_ENGINEERING'
ete_path = f'{results}/END_TO_END'

classifiers = ['knn', 'svm', 'xgb']
subject_ids = ['11', '12', '13', '14', '15', '16', '17', '18', '20', '21']
def metrics_for(path):
    for device_signal in os.listdir(path):
      [device, signal] = device_signal.split('_')
      print(device, signal)
      for classifier in classifiers:
        best_f1 = 0
        best_model = ''
        for metrics in [f'{path}/{device_signal}/test/{classifier}{id}_df_metrics.csv' for id in subject_ids]:
          df = pd.read_csv(metrics).head()
          precision = float(df['precision'].iloc[0])
          recall = float(df['recall'].iloc[0])
          duration = float(df['duration'].iloc[0])
          f1 = float(df['f1'].iloc[0])
          if f1 > best_f1:
            best_f1 = f1
            best_model = metrics
        print(f'  classifier: {classifier}')
        print(f'    f1: {best_f1}, best_model: {best_model}')

metrics_for(fee_path)
