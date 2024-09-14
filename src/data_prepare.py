import os
import re
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
      return pd[list(filter(None, [col if wave in col else None for wave in eeg_waves for col in pd.columns]))]

def merge_experiments(path):
  print("merging experiments")
  for file in os.listdir(path):
    os.mkdir(os.path.join(path, file, 'baseline'))
    os.mkdir(os.path.join(path, file, 'cognitive_load'))
    for label in os.listdir(os.path.join(path, file, 'pre')):
      for f in os.listdir(os.path.join(path, file, 'pre', label)):
        pre_file = pd.read_csv(os.path.join(path, file, 'pre', label, f))
        if 'eeg' in f:
          pre_file = filter_eeg_columns(pre_file)
        elif 'empatica_bvp.csv' == f:
          pre_file.rename(columns={"bvp": "ppg"}, inplace=True)
        elif 'samsung_bvp.csv' == f:
          pre_file.rename(columns={"PPG GREEN": "ppg"}, inplace=True)
        pre_file.to_csv(os.path.join(path,file, label, f), index=False)
        os.remove(os.path.join(path, file, "pre", label, f))
      os.rmdir(os.path.join(path, file, 'pre', label))
    os.rmdir(os.path.join(path, file, 'pre'))

    if os.path.exists(os.path.join(path, file, 'post')):
      for label in os.listdir(os.path.join(path, file, 'post')):
        for f in os.listdir(os.path.join(path, file, 'post',label)):
          if os.path.exists(os.path.join(path, file, 'post', label, f)):
            post_file = pd.read_csv(os.path.join(path, file, 'post', label, f))
            if 'eeg' in f:
              post_file = filter_eeg_columns(post_file)
            elif 'empatica_bvp.csv' == f:
              post_file.rename(columns={"bvp": "ppg"}, inplace=True)
            elif 'samsung_bvp.csv' == f:
              post_file.rename(columns={"PPG GREEN": "ppg"}, inplace=True)
            concat = [post_file]
            if os.path.exists(os.path.join(path, file, label, f)):
              pre_file = pd.read_csv(os.path.join(path, file, label, f))
              concat.append(pre_file)
            pd.concat(concat).to_csv(os.path.join(path, file, label, f), index=False)
            os.remove(os.path.join(path, file, 'post', label, f))
        os.rmdir(os.path.join(path, file, 'post', label))
      os.rmdir(os.path.join(path, file, 'post'))

def join_ys(path):
  print("join y")
  for participant in os.listdir(path):
    for csv in os.listdir(os.path.join(path, participant, 'baseline')):
      df = pd.read_csv(os.path.join(path, participant, 'baseline', csv))
      df = df.assign(y=np.zeros(len(df), dtype=int))
      os.remove(os.path.join(path, participant, 'baseline', csv))
      df.to_csv(os.path.join(path, participant, csv), index=False)
    os.rmdir(os.path.join(path, participant, 'baseline'))

    for csv in os.listdir(os.path.join(path, participant, 'cognitive_load')):
      df = pd.read_csv(os.path.join(path, participant, 'cognitive_load', csv))
      df = df.assign(y=np.ones(len(df), dtype=int))
      if os.path.exists(os.path.join(path, participant, csv)):
        df2 = pd.read_csv(os.path.join(path, participant, csv));
        df = pd.concat([df, df2])
      df.to_csv(os.path.join(path, participant, csv), index=False)
      os.remove(os.path.join(path, participant, 'cognitive_load', csv))
    os.rmdir(os.path.join(path, participant, 'cognitive_load'))

def rename_ppg_sensor(path):
  for participant in os.listdir(path):
    for csv in os.listdir(os.path.join(path, participant)):
      if "bvp" in csv:
        new_csv = re.sub("bvp", "ppg", csv)
        os.rename(os.path.join(path, participant, csv), os.path.join(path, participant, new_csv))

def prepare_data(path):
  print("remove survey measurements")
  remove_survey_measurements(path)
  print("remove unused files")
  remove_unused_files_recursive(path)
  print("merge pre and post experiments")
  merge_experiments(path)
  print("join baseline and cognitive efort")
  join_ys(path)
  print("rename sensor to ppg")
  rename_ppg_sensor(path)
