from src.data_prepare import prepare_data, join_ys, rename_ppg_sensor
import os 

survey_path = "{0}/survey_gamification".format(os.getcwd())
prepare_data(survey_path)

# join_ys(survey_path)
# print("rename sensor to ppg")
# rename_ppg_sensor(survey_path)
