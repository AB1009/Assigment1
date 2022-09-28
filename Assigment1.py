import pandas as pd
import numpy as np
from pandas import read_csv
from matplotlib import pyplot
# Loading dataset
data = pd.read_csv(r'D:\Data1 (1).csv')
X = np.array(data[['T','P','TC','SV']])
y = np.array(data['Idx'])

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

ridge = Ridge(alpha = 1.0)
mymodel = ridge.fit(X_test, y_test)

lasso =  Lasso(alpha =0.0001)
mymodel1 = lasso.fit(X_test, y_test)

elasticnet = ElasticNet(alpha = 0.01)
mymodel2 = elasticnet.fit(X_test, y_test)

from sklearn.model_selection import cross_val_score

RidgeCV = cross_val_score(mymodel, X_test, y_test, scoring = 'r2', cv = 10)
LassoCV = cross_val_score(mymodel1, X_test, y_test, scoring = 'r2', cv = 10)
ElasticNetCV = cross_val_score(mymodel2, X_test, y_test, scoring = 'r2', cv = 10)

#K-Fold CV
print('K-fold CV (Ridge) -',RidgeCV)
print('Mean -',np.mean(RidgeCV))
print('K-fold CV (Lasso) -',LassoCV)
print('Mean -',np.mean(LassoCV))
print('K-fold CV (Elastic-Net) -',ElasticNetCV)
print('Mean -',np.mean(ElasticNetCV))