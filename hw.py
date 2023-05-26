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


