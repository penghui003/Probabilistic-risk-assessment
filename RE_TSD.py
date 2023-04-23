import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures, RobustScaler
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error,mean_absolute_percentage_error,r2_score

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('SFEM RESULTS.csv', header=1)
features = tpot_data.drop(['y1','y2','y3'], axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['y2'], test_size=0.2, random_state=0)
# Average CV score on the training set was: -0.504191653444035
model2 = exported_pipeline = make_pipeline(
    RobustScaler(),
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    ElasticNetCV(l1_ratio=0.2, tol=0.0001)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(model2.steps, 'random_state', 1)
re_tsd=model2.fit(training_features, training_target)

# def re_tsd(x):
#     pre0=RobustScaler().fit(training_features)
#     pre1=PolynomialFeatures(degree=2, include_bias=False, interaction_only=False).fit(pre0.transform(training_features))
#     pre2=StackingEstimator(estimator=LassoLarsCV(normalize=True)).fit(pre1.transform(pre0.transform(training_features)),training_target)
#     pre=ElasticNetCV(l1_ratio=0.2, tol=0.0001).fit(pre2.transform(pre1.transform(pre0.transform(training_features))),training_target)
#     result=pre.predict(pre2.transform(pre1.transform(pre0.transform(x))))
#     return result

# y2_pred=re_tsd(testing_features.values)
# y2_pred_train=re_tsd(training_features.values)

# print("rmse \t:", sqrt(mean_squared_error(testing_target, y2_pred)))
# print("mae \t:", mean_absolute_error(testing_target, y2_pred))
# print("mape \t:", mean_absolute_percentage_error(testing_target, y2_pred))
# print("r2 \t:", r2_score(testing_target, y2_pred))

# print("rmse_train \t:", sqrt(mean_squared_error(training_target, y2_pred_train)))
# print("mae_train \t:", mean_absolute_error(training_target, y2_pred_train))
# print("mape \t:", mean_absolute_percentage_error(training_target, y2_pred_train))
# print("r2_train \t:", r2_score(training_target, y2_pred_train))