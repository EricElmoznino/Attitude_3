import Attitude_3
import Helpers

att_model = Attitude_3.Model(Helpers.Configuration(epochs=50, batch_size=10, seq_size=2), 80, 80)
# att_model.train('/Users/Eric/ML_data/Attitude_2/train_data',
#                 '/Users/Eric/ML_data/Attitude_2/validation_data',
#                 '/Users/Eric/ML_data/Attitude_2/test_data')
# predictions = att_model.predict('/Users/Eric/ML_data/Attitude_2/prediction_data')
# att_model.train('./data/train_data',
#                 './data/validation_data',
#                 './data/test_data')
# predictions = att_model.predict('./data/prediction_data')
att_model.train('../Attitude_2/data/train_data')
# att_model.train('/Users/Eric/ML_data/Attitude_2/train_data')
