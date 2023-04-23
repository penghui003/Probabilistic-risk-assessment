#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error,mean_absolute_percentage_error,r2_score
from math import sqrt
from pandas import Series, DataFrame
import GPy
from sklearn.ensemble import RandomForestRegressor
import tpot
from xgboost import XGBRegressor
#%%
data = pd.read_csv("SFEM RESULTS.csv", header=1)

Y = data.loc[:,['y1','y2','y3']]
X= data.drop(['y1','y2','y3'],axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
y1_train=y_train['y1']
y1_test=y_test['y1']
y2_train=y_train['y2']
y2_test=y_test['y2']
y3_train=y_train['y3']
y3_test=y_test['y3']

#%%
cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)
model1=tpot.TPOTRegressor(generations=10, population_size=100, scoring='neg_mean_squared_error', cv=cv, random_state=1, n_jobs=-1, verbosity=3)

# model1=RandomForestRegressor()
# model1=XGBRegressor(max_depth=3,min_child_weight=7,subsample=0.5,colsample_bytree=0.5,reg_alpha=0,reg_lambda=0,gamma=0.8)

regr1=model1.fit(x_train, y1_train)
regr1.export('model1.py')
y1_pred=regr1.predict(x_test.values)
y1_pred_train=regr1.predict(x_train.values)

print("rmse \t:", sqrt(mean_squared_error(y1_test, y1_pred)))
print("mae \t:", mean_absolute_error(y1_test, y1_pred))
print("mape \t:", mean_absolute_percentage_error(y1_test, y1_pred))
print("r2 \t:", r2_score(y1_test, y1_pred))

print("rmse_train \t:", sqrt(mean_squared_error(y1_train, y1_pred_train)))
print("mae_train \t:", mean_absolute_error(y1_train, y1_pred_train))
print("mape \t:", mean_absolute_percentage_error(y1_train, y1_pred_train))
print("r2_train \t:", r2_score(y1_train, y1_pred_train))
# print(getattr(regr1,"pareto_front_fitted_pipelines_"))
# print(getattr(regr1,"fitted_pipeline_"))
#%%
cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=0)
model2=tpot.TPOTRegressor(generations=10, population_size=100, scoring='neg_mean_squared_error', cv=cv, verbosity=3, random_state=1, n_jobs=-1)

# # model2=RandomForestRegressor()
# model2=XGBRegressor(max_depth=3,min_child_weight=5,subsample=0.5,colsample_bytree=0.5,reg_alpha=0,reg_lambda=0,gamma=0.5)

regr2=model2.fit(x_train, y2_train)
regr2.export('model2.py')
y2_pred=regr2.predict(x_test.values)
y2_pred_train=regr2.predict(x_train.values)

print("rmse \t:", sqrt(mean_squared_error(y2_test, y2_pred)))
print("mae \t:", mean_absolute_error(y2_test, y2_pred))
print("mape \t:", mean_absolute_percentage_error(y2_test, y2_pred))
print("r2 \t:", r2_score(y2_test, y2_pred))

print("rmse_train \t:", sqrt(mean_squared_error(y2_train, y2_pred_train)))
print("mae_train \t:", mean_absolute_error(y2_train, y2_pred_train))
print("mape \t:", mean_absolute_percentage_error(y2_train, y2_pred_train))
print("r2_train \t:", r2_score(y2_train, y2_pred_train))

# print(getattr(regr2,"pareto_front_fitted_pipelines_"))
# print(getattr(regr2,"fitted_pipeline_"))

#%%
cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)
model3=tpot.TPOTRegressor(generations=10, population_size=100, scoring='neg_mean_squared_error', cv=cv, verbosity=3, random_state=1, n_jobs=-1)

# model3=RandomForestRegressor(n_estimators=100,max_depth=10, min_samples_split=5,min_samples_leaf=5,oob_score=True, random_state=1)
# model3=XGBRegressor(max_depth=3,min_child_weight=9,subsample=0.5,colsample_bytree=0.5,reg_alpha=0,reg_lambda=0,gamma=0.8)

regr3=model3.fit(x_train, y3_train)
regr3.export('model3.py')
y3_pred=regr3.predict(x_test.values)
y3_pred_train=regr3.predict(x_train.values)
print("rmse \t:", sqrt(mean_squared_error(y3_test, y3_pred)))
print("mae \t:", mean_absolute_error(y3_test, y3_pred))
print("mape \t:", mean_absolute_percentage_error(y3_test, y3_pred))
print("r2 \t:", r2_score(y3_test, y3_pred))

print("rmse_train \t:", sqrt(mean_squared_error(y3_train, y3_pred_train)))
print("mae_train \t:", mean_absolute_error(y3_train, y3_pred_train))
print("mape \t:", mean_absolute_percentage_error(y3_train, y3_pred_train))
print("r2_train \t:", r2_score(y3_train, y3_pred_train))
# print(getattr(regr3,"pareto_front_fitted_pipelines_"))
# print(getattr(regr3,"fitted_pipeline_"))
