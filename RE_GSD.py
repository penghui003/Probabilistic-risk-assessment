import numpy as np
import pandas as pd
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from tpot.builtins import ZeroCount
from tpot.export_utils import set_param_recursive
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error,mean_absolute_percentage_error,r2_score

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('SFEM RESULTS.csv', header=1)
features = tpot_data.drop(['y1','y2','y3'], axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['y3'], test_size=0.2, random_state=0)
# Average CV score on the training set was: -1.488134107021547
model3 =  make_pipeline(
    StandardScaler(),
    ZeroCount(),
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    LassoLarsCV(normalize=False)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(model3.steps, 'random_state', 1)
re_gsd=model3.fit(training_features, training_target)


# def re_gsd(x):
#     pre0=StandardScaler().fit(training_features)
#     pre1=ZeroCount().fit(pre0.transform(training_features))
#     pre2=PolynomialFeatures(degree=2, include_bias=False, interaction_only=False).fit(pre1.transform(pre0.transform(training_features)),training_target)
#     pre=LassoLarsCV(normalize=False).fit(pre2.transform(pre1.transform(pre0.transform(training_features))),training_target)
#     result=pre.predict(pre2.transform(pre1.transform(pre0.transform(x))))
#     return result

# y3_pred=re_gsd(testing_features.values)
# y3_pred_train=re_gsd(training_features.values)

# print("rmse \t:", sqrt(mean_squared_error(testing_target, y3_pred)))
# print("mae \t:", mean_absolute_error(testing_target, y3_pred))
# print("mape \t:", mean_absolute_percentage_error(testing_target, y3_pred))
# print("r2 \t:", r2_score(testing_target, y3_pred))

# print("rmse_train \t:", sqrt(mean_squared_error(training_target, y3_pred_train)))
# print("mae_train \t:", mean_absolute_error(training_target, y3_pred_train))
# print("mape \t:", mean_absolute_percentage_error(training_target, y3_pred_train))
# print("r2_train \t:", r2_score(training_target, y3_pred_train))

