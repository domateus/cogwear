from src.data_prepare import prepare_data
import os 

survey_path = "{0}/survey_gamification".format(os.getcwd())
pilot = "{0}/pilot".format(os.getcwd())
prepare_data(pilot, True)

