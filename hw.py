import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
from numpy import where

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 130)
pd.options.display.float_format = '{:,.3f}'.format

###1. read data 
data = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')


###2. original 결측치(nan) 제거 
data = data.dropna(subset=['original'], how='any', axis=0)
data_test = data_test.dropna(subset=['original'], how='any', axis=0)


###3. imputation 
#ref from https://www.kaggle.com/code/inversion/get-started-with-mean-imputation
from sklearn.impute import SimpleImputer
import numpy as np

imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
data[:] = imp.fit_transform(data)

#imputation for test data
data_test[:] = imp.fit_transform(data_test)


###4. replace -4,-5 => -3
data = data.replace(-5, -3)
data = data.replace(-4, -3)

data_test = data_test.replace(-5, -3)
data_test = data_test.replace(-4, -3)


###5. data balancing: under_sampling

#ref from: https://medium.com/grabngoinfo/four-oversampling-and-under-sampling-methods-for-imbalanced-classification-using-python-7304aedf9037
#random undersample for imbalanced dataset
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt

#define dataset for data balancing(under_sampling)
x = data.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
    21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]] 
y = data['timestamp(day)'] #index=34

rus = RandomUnderSampler(random_state=42)
x, y = rus.fit_resample(x, y)

#test data balancing: under_sampling
x_test = data_test.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
    21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]] 
y_test = data_test['timestamp(day)']

rus = RandomUnderSampler(random_state=42)
x_test, y_test = rus.fit_resample(x_test, y_test)


###7. part 1 answer 

###1) patient’s count 
print('\n#patient count')
print(data['timestamp(day)'].count()) 


###2) the mean and median value of the label (day)
'''
print('\n#timestamp(day) mean')
print(x['timestamp(day)'].mean())
print('\n#timestamp(day) median')
print(x['timestamp(day)'].median())
'''
###3) Perform EDA and calculate the statistics of the dataset: 
 #   mean, std, correlations among features, etc. 
 #   (e.g.There are 34 features and you have to find the correlations 
 #   among each feature (34 by 34 correlation matrix)).

'''
print('\n# x.describe')
print(x.describe())
print('\n# x.corr')
print(x.corr())
'''

'''
corr_with_day = x.corr()[['timestamp(day)']]
print('\n#corr_with_day')
#print(corr_with_day)
'''

'''
############################# part 2 start 
###8. define dataset after data balancing

#x_train = x.copy().iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
#    21,22,23,24,25,26,27,28,29,30,31,32,33]] 
#=> 5,6,8,16,17,18,19,20,21,23,25,27,28,29,30 제외: by SMOTE where corr < 0.01
#x_train = x.copy().iloc[:, [0,1,2,3,4,7,9,10,11,12,13,14,15,
#    22,24,26,29,31,32,33]] 
#=> 8,16,17,18,21,29,30,31 index 제외: by RandomUnderSampler where corr < 0.005
x_train = x.copy().iloc[:, [0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,19,20,
    22,23,24,25,26,27,28,32,33]] 

y_train = x['timestamp(day)'] #index=34

print('\n#x_train.describe() after balancing')
print(x_train.describe())

print('#count y_train after balancing')
print(y_train.count())

#define dataset test after data balancing
y_test = x_test['timestamp(day)'] #index=34

#x_test = x_test.copy().iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
#    21,22,23,24,25,26,27,28,29,30,31,32,33]] 
#=>
#x_test = x_test.copy().iloc[:, [0,1,2,3,4,7,9,10,11,12,13,14,15,
#    22,24,26,29,31,32,33]] 
#=> 8,16,17,18,21,29,30,31 index 제외: by RandomUnderSampler where corr < 0.005
x_test = x_test.copy().iloc[:, [0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,19,20,
    22,23,24,25,26,27,28,32,33]] 

print('\n#count y_test after balancing')
print(y_test.count())


###9. train: random forest model

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

#rf = RandomForestClassifier()
rf = RandomForestClassifier(max_depth=9, max_features='sqrt', max_leaf_nodes=9, n_estimators=150)
#rf = RandomForestClassifier(max_depth=9, max_features="log2", max_leaf_nodes=9,n_estimators=25)
model = rf.fit(x_train, y_train)
pred = model.predict(x_test)

print('#classification_report(y_test, pred)')
print(classification_report(y_test, pred))

###0. parameter tunning 

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from datetime import datetime

now = datetime.now()
print('#now')
print(now)

param_grid = {
    'n_estimators': [25, 50, 100, 150],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [3, 6, 9],
    'max_leaf_nodes': [3, 6, 9],
}

grid_search = GridSearchCV(RandomForestClassifier(),
                           param_grid=param_grid)
grid_search.fit(x_train, y_train)
print('#grid_search.best_estimator_')
print(grid_search.best_estimator_)
#=> 결과 
#RandomForestClassifier(max_depth=9, max_features='sqrt', max_leaf_nodes=9)

now = datetime.now()
print('#now')
print(now)



###9. train: xgboost example

from xgboost import XGBClassifier
from sklearn.metrics import classification_report

y_train = y_train.replace(-3, 3)
y_train = y_train.replace(-2, 2)
y_train = y_train.replace(-1, 1)

y_test = y_test.replace(-3, 3)
y_test = y_test.replace(-2, 2)
y_test = y_test.replace(-1, 1)

model = XGBClassifier()
xgb_model = model.fit(x_train, y_train, eval_metric='logloss')
#xgb_model = model.fit(x_train, y_train, early_stopping_rounds=100, eval_metric='logloss')

y_pred = xgb_model.predict(x_test)

print(classification_report(y_test, y_pred))


###10. ROC curve 

#ref url: https://continuous-development.tistory.com/m/172
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

#print('\n#type(y_test)')
#print(type(y_test))
y_test_np = y_test.to_numpy()
#pred_np = pred.to_numpy()
fprs, tprs, thresholds = roc_curve(y_test_np, pred, pos_label=-3)

plt.figure(figsize=(15,5))
# 대각선
plt.plot([0,1],[0,1],label='STR')
# ROC
plt.plot(fprs,tprs,label='ROC')

plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.grid()
plt.show()


###2. patient’s count 

d_patient = data.where((data['timestamp(day)'] == 0) & (data['timestamp(hr)'] == 0))
print('\n#patient count')
print(d_patient['timestamp(day)'].count()) #4156
'''

'''
###6. SMOTE over_sampling

from imblearn.over_sampling import RandomOverSampler, SMOTE
from collections import Counter

x = data.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
    21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]] 
y = data['timestamp(day)'] #index=34

x_test = data_test.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
    21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]] 
y_test = data_test['timestamp(day)']

smote = SMOTE(random_state=42)
x, y= smote.fit_resample(x,y)

x_test, y_test= smote.fit_resample(x_test,y_test)

counter = Counter(y)
print('\n#Counter y after smote')
print(sorted(Counter(y).items()))

print('\n#corr after smote')
print(x.corr())

'''
