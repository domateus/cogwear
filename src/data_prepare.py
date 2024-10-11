import os
import shutil
import re
from time import time
from keras.src.ops import dtype
import pandas as pd
import numpy as np

survey_path = "{0}/survey_gamification".format(os.getcwd())

def remove_survey_measurements(path):
  for file in os.listdir(path):
    if os.path.isdir(os.path.join(path, file)) and file == 'survey':
      for measurement in os.listdir(os.path.join(path, file)):
        os.remove(os.path.join(path, file, measurement))
      os.rmdir(os.path.join(path, file))
    elif os.path.isdir(os.path.join(path, file)):
      remove_survey_measurements(os.path.join(path, file))

unused_files = ['html' , 'temp' , 'stroop' , 'questionnaire' , 'responses']
def remove_unused_files_recursive(data_path):
  for file in os.listdir(data_path):
    if any(unused in file for unused in unused_files):
      os.remove(os.path.join(data_path, file))
    elif os.path.isdir(os.path.join(data_path, file)):
      remove_unused_files_recursive(os.path.join(data_path, file))

eeg_waves = ['Alpha', 'Beta', 'Gamma', 'Theta', 'Delta']
def filter_eeg_columns(pd):
      result = pd[list(filter(None, [col if wave in col or col == 'y' else None for wave in eeg_waves for col in pd.columns]))]
      return result.dropna()

def get_remove_window_size(filename: str) -> int:
  if 'eeg' in filename:
    return 4*256
  if 'eda' in filename:
    return 4*4
  if 'samsung' in filename:
    return 4*25
  return 4*64

files = ['samsung_bvp.csv', 'empatica_bvp.csv', 'empatica_eda.csv', 'muse_eeg.csv']
def df(source_path, subject, day, label, file):
  path = ''
  if day is None:
    path = os.path.join(source_path, subject, label, file)
  else:
    path = os.path.join(source_path, subject, day, label, file)
  if os.path.exists(path):
    return pd.read_csv(path)
  return pd.DataFrame()

def merge_files(path):
  print(f'merging files at: {path}')
  for subject in os.listdir(path):
    print(f'  SUBJECT: {subject}')
    for file in files:
      start_time = time()
      pre_baseline = df(path, subject, 'pre', 'baseline', file)
      pre_cl = df(path, subject, 'pre', 'cognitive_load', file)
      post_baseline = df(path, subject, 'post', 'baseline', file)
      post_cl = df(path, subject, 'post', 'cognitive_load', file)

      # given that the experiment start with the baseline measurement and ends with the cognitive load
      # the beggining of the baseline is removed and the end of cognitive load is removed
      ws = get_remove_window_size(file)
      pre_baseline = pre_baseline.iloc[ws:]
      post_baseline = post_baseline.iloc[ws:]
      yb_pre = np.zeros(len(pre_baseline), dtype=int)
      pre_baseline = pre_baseline.assign(y=yb_pre)
      yb_post = np.zeros(len(post_baseline), dtype=int)
      post_baseline = post_baseline.assign(y=yb_post)

      pre_cl = pre_cl.iloc[:-ws]
      post_cl = post_cl.iloc[:-ws]
      yc_pre = np.ones(len(pre_cl), dtype=int)
      pre_cl = pre_cl.assign(y=yc_pre)
      yc_post = np.ones(len(post_cl), dtype=int)
      post_cl = post_cl.assign(y=yc_post)

      data = pd.concat([pre_baseline, pre_cl, post_baseline, post_cl])

      if 'eeg' in file:
        data = filter_eeg_columns(data)

      elif 'empatica_bvp.csv' == file:
        file = re.sub("bvp", "ppg", file)
        data.rename(columns={"bvp": "ppg"}, inplace=True)
      elif 'samsung_bvp.csv' == file:
        file = re.sub("bvp", "ppg", file)
        data.rename(columns={"PPG GREEN": "ppg"}, inplace=True)
      
      data.to_csv(os.path.join(path, subject, file), index=False)

      took = f'{round(time() - start_time, 2)}"'
      print(f'    file: {file} - {took}')
    if os.path.exists(os.path.join(path, subject, 'pre')):
      shutil.rmtree(os.path.join(path, subject, 'pre'))
    if os.path.exists(os.path.join(path, subject, 'post')):
      shutil.rmtree(os.path.join(path, subject, 'post'))

def prepare_data(path):
  print("remove survey measurements")
  remove_survey_measurements(path)
  print("remove unused files")
  remove_unused_files_recursive(path)
  merge_files(path)
