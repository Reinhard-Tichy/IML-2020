import pandas as pd 
import os 
import numpy as np 

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
#from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV


class XGboost_model():
    def __init__(self, need_scale=False) -> None:
        super().__init__()
        self.lr = XGBRegressor(
                learning_rate=0.05,max_depth=3,n_estimators=100,random_state=None)
        self.dropAttr = ['charges']
        if need_scale:
            self.scaler = StandardScaler()
        else :
            self.scaler = None

    def train(self, data : pd.DataFrame):
        x = data.drop(self.dropAttr, axis = 1)
        if self.scaler : x = self.scaler.fit_transform(x)
        y = data.charges

        x_train,x_test,y_train,y_test = train_test_split(x,y, random_state = None)
        self.lr.fit(x_train,y_train)

        print("train_score: {:.15f}".format(self.lr.score(x_train, y_train)))
        print("val_score: {:.15f}".format(self.lr.score(x_test,y_test)))

    def test(self, data : pd.DataFrame):
        test_x = data.drop(self.dropAttr, axis = 1)
        if self.scaler : test_x = self.scaler.transform(test_x)
        test_y = self.lr.predict(test_x)
        return test_y

class GBM_model():
    def __init__(self, n_estimators=100, need_scale=False) -> None:
        super().__init__()
        self.lr = GradientBoostingRegressor(
                n_estimators=n_estimators, learning_rate=0.05, random_state = 0
        )
        self.dropAttr = ['charges']
        if need_scale:
            self.scaler = StandardScaler()
        else :
            self.scaler = None

    def train(self, data : pd.DataFrame):
        x = data.drop(self.dropAttr, axis = 1)
        if self.scaler : x = self.scaler.fit_transform(x)
        y = data.charges

        x_train,x_test,y_train,y_test = train_test_split(x,y, random_state = 2)
        self.lr.fit(x_train,y_train)

        print("train_score: {:.15f}".format(self.lr.score(x_train, y_train)))
        print("val_score: {:.15f}".format(self.lr.score(x_test,y_test)))
        return self.lr.score(x_train, y_train), self.lr.score(x_test,y_test)
        
    def test(self, data : pd.DataFrame):
        test_x = data.drop(self.dropAttr, axis = 1)
        if self.scaler : test_x = self.scaler.transform(test_x)
        test_y = self.lr.predict(test_x)
        return test_y
    

class Bagging_model():
    def __init__(self, n_estimators=60 , need_scale=False) -> None:
        super().__init__()
        self.base_lr = MLPRegressor(
    hidden_layer_sizes=(6,8,4),  activation='relu', solver='lbfgs', alpha=0.0001, batch_size='auto',
    learning_rate='adaptive', learning_rate_init=0.001, power_t=0.5, max_iter=8000, shuffle=True,
    random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False,beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.lr = BaggingRegressor(base_estimator=self.base_lr, n_estimators=n_estimators,
            max_samples=1.0, bootstrap=True, random_state=2, n_jobs=-1)
        self.dropAttr = ['charges']
        if need_scale:
            self.scaler = StandardScaler()
        else :
            self.scaler = None
    
    def train(self, data : pd.DataFrame):
        x = data.drop(self.dropAttr, axis = 1)
        if self.scaler : x = self.scaler.fit_transform(x)
        y = data.charges

        x_train,x_test,y_train,y_test = train_test_split(x,y, random_state = 2)
        self.lr.fit(x_train,y_train)

        print("train_score: {:.15f}".format(self.lr.score(x_train, y_train)))
        print("val_score: {:.15f}".format(self.lr.score(x_test,y_test)))
        return self.lr.score(x_train, y_train), self.lr.score(x_test,y_test)

    def test(self, data : pd.DataFrame):
        test_x = data.drop(self.dropAttr, axis = 1)
        if self.scaler : test_x = self.scaler.transform(test_x)
        test_y = self.lr.predict(test_x)
        return test_y


class MLP_model():
    def __init__(self, need_scale=False) -> None:
        super().__init__()
        self.lr = MLPRegressor(
    hidden_layer_sizes=(6,8,4),  activation='relu', solver='lbfgs', alpha=0.0001, batch_size='auto',
    learning_rate='adaptive', learning_rate_init=0.001, power_t=0.5, max_iter=8000, shuffle=True,
    random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False,beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.dropAttr = ['charges']
        if need_scale:
            self.scaler = StandardScaler()
        else :
            self.scaler = None

    def train(self, data : pd.DataFrame):
        x = data.drop(self.dropAttr, axis = 1)
        if self.scaler : x = self.scaler.fit_transform(x)
        y = data.charges

        x_train,x_test,y_train,y_test = train_test_split(x,y, random_state = 2)
        self.lr.fit(x_train,y_train)

        print("train_score: {:.15f}".format(self.lr.score(x_train, y_train)))
        print("val_score: {:.15f}".format(self.lr.score(x_test,y_test)))

    def test(self, data : pd.DataFrame):
        test_x = data.drop(self.dropAttr, axis = 1)
        if self.scaler : test_x = self.scaler.transform(test_x)
        test_y = self.lr.predict(test_x)
        return test_y

class LR_model():
    def __init__(self, need_scale=False) -> None:
        self.lr = LinearRegression()
        self.dropAttr = ['charges']
        if need_scale:
            self.scaler = StandardScaler()
        else :
            self.scaler = None
    def train(self, data : pd.DataFrame):
        x = data.drop(self.dropAttr, axis = 1)
        if self.scaler : x = self.scaler.fit_transform(x)
        y = data.charges

        x_train,x_test,y_train,y_test = train_test_split(x,y, random_state = 0)
        self.lr.fit(x_train,y_train)

        print("train_score: {:.15f}".format(self.lr.score(x_train, y_train)))
        print("val_score: {:.15f}".format(self.lr.score(x_test,y_test)))
    
    def test(self, data : pd.DataFrame):
        test_x = data.drop(self.dropAttr, axis = 1)
        if self.scaler : test_x = self.scaler.transform(test_x)
        test_y = self.lr.predict(test_x)
        return test_y

class PolyLR_model():
    def __init__(self, need_scale = False) -> None:
        self.lr = LinearRegression()
        self.dropAttr = ['charges']
        self.poly = PolynomialFeatures(degree = 4)
        if need_scale:
            self.scaler = StandardScaler()
        else :
            self.scaler = None

    def train(self, data : pd.DataFrame):
        x = data.drop(self.dropAttr, axis = 1)
        y = data.charges
        if self.scaler : x = self.scaler.fit_transform(x)
        x_quad = self.poly.fit_transform(x)
        x_train,x_test,y_train,y_test = train_test_split(x_quad, y, random_state = 0)
        self.lr.fit(x_train,y_train)

        print("train_score: {:.15f}".format(self.lr.score(x_train, y_train)))
        print("val_score: {:.15f}".format(self.lr.score(x_test,y_test)))

    def test(self, data : pd.DataFrame):
        test_x = data.drop(self.dropAttr, axis = 1)
        if self.scaler : test_x = self.scaler.transform(test_x)
        test_x_quad = self.poly.transform(test_x)
        test_y = self.lr.predict(test_x_quad)
        return test_y

class forest_model():
    def __init__(self, n_estimators=50 , need_scale = False) -> None:
        self.forest = RandomForestRegressor(n_estimators = n_estimators,
                              criterion = 'mse', bootstrap = True,
                              random_state = 1,
                              max_features='auto', n_jobs = -1)
        self.dropAttr = ['charges']
        self.poly = PolynomialFeatures(degree = 1)
    
    def train(self, data : pd.DataFrame):
        x = data.drop(self.dropAttr, axis = 1)
        y = data.charges
        
        x = self.poly.fit_transform(x)
        x_train,x_test,y_train,y_test = train_test_split(x,y, random_state = 2)
        self.forest.fit(x_train,y_train)
        forest_train_pred = self.forest.predict(x_train)
        forest_test_pred = self.forest.predict(x_test)
        print("train_score: {:.15f}".format(self.forest.score(x_train, y_train)))
        print("val_score: {:.15f}".format(self.forest.score(x_test,y_test)))

        mse_train = mse(y_train,forest_train_pred)
        mse_test = mse(y_test,forest_test_pred)
        print('MSE train data: %.3f, MSE test data: %.3f' % (mse_train, mse_test))
        return self.forest.score(x_train, y_train), self.forest.score(x_test,y_test)

    def test(self, data : pd.DataFrame):
        test_x = data.drop(self.dropAttr, axis = 1)
        test_x = self.poly.transform(test_x)
        test_y = self.forest.predict(test_x)
        return test_y

class SVR_model():
    def __init__(self, need_scale = False) -> None:
        self.lr = SVR(kernel = 'rbf')
        self.dropAttr = ['charges']
        if need_scale:
            self.scaler = StandardScaler()
        else :
            self.scaler = None

    def train(self, data : pd.DataFrame):
        x = data.drop(self.dropAttr, axis = 1)
        if self.scaler : x = self.scaler.fit_transform(x)
        y = data.charges

        x_train,x_test,y_train,y_test = train_test_split(x,y, random_state = None)
        self.lr.fit(x_train,y_train)

        print("train_score: {:.15f}".format(self.lr.score(x_train, y_train)))
        print("val_score: {:.15f}".format(self.lr.score(x_test,y_test)))
    
    def test(self, data : pd.DataFrame):
        test_x = data.drop(self.dropAttr, axis = 1)
        if self.scaler : test_x = self.scaler.transform(test_x)
        test_y = self.lr.predict(test_x)
        return test_y  
    

