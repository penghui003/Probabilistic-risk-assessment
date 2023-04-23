import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler
from tpot.builtins import OneHotEncoder, StackingEstimator
from tpot.export_utils import set_param_recursive
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error,mean_absolute_percentage_error,r2_score

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('SFEM RESULTS.csv', header=1)
features = tpot_data.drop(['y1','y2','y3'], axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['y1'], test_size=0.2, random_state=0)
# Average CV score on the training set was: -19.515533795453663
model1 = make_pipeline(
    StandardScaler(),
    OneHotEncoder(minimum_fraction=0.15, sparse=False, threshold=10),
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    GradientBoostingRegressor(alpha=0.99, learning_rate=0.1, loss="huber", max_depth=2, max_features=0.8, min_samples_leaf=15, min_samples_split=4, n_estimators=100, subsample=1.0)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(model1.steps, 'random_state', 1)
re_lsp=model1.fit(training_features, training_target)

# def re_lsp(x):
#     pre0=StandardScaler().fit(training_features)
#     pre1=OneHotEncoder(minimum_fraction=0.15, sparse=False, threshold=10).fit(pre0.transform(training_features))
#     pre2=StackingEstimator(estimator=LassoLarsCV(normalize=True)).fit(pre1.transform(pre0.transform(training_features)),training_target)
#     pre= GradientBoostingRegressor(alpha=0.99, learning_rate=0.1, loss="huber", max_depth=2, max_features=0.8, min_samples_leaf=15, min_samples_split=4, n_estimators=100, subsample=1.0).fit(pre2.transform(pre1.transform(pre0.transform(training_features))),training_target)
#     result=pre.predict(pre2.transform(pre1.transform(pre0.transform(x))))
#     return result

# y1_pred=re_lsp(testing_features.values)
# y1_pred_train=re_lsp(training_features.values)

# print("rmse \t:", sqrt(mean_squared_error(testing_target, y1_pred)))
# print("mae \t:", mean_absolute_error(testing_target, y1_pred))
# print("mape \t:", mean_absolute_percentage_error(testing_target, y1_pred))
# print("r2 \t:", r2_score(testing_target, y1_pred))

# print("rmse_train \t:", sqrt(mean_squared_error(training_target, y1_pred_train)))
# print("mae_train \t:", mean_absolute_error(training_target, y1_pred_train))
# print("mape \t:", mean_absolute_percentage_error(training_target, y1_pred_train))
# print("r2_train \t:", r2_score(training_target, y1_pred_train))


