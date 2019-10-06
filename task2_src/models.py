import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from utils import filter_outlier, dtye_detection
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import os
import pickle


class Models():
    
    def __init__(self, df, xgb_params, lgbm_params, cat_params, n_fold=5):
        
        self.df = df
        self.xgb_params = xgb_params
        self.lgbm_params = lgbm_params
        self.cat_params = cat_params
        self.n_folsd = n_fold
        self.target = 'price'
        self.xgb_df = pd.get_dummies(self.df)
    

        
    def _split_X_y(self, df):
        
        kf = StratifiedKFold(n_splits=5, random_state=100)
        X_data, y_data = df.drop((self.target), axis=1), df[self.target]
        bins = np.linspace(int(y_data.min()), int(y_data.max())+1, 10)
        y_binned = np.digitize(y_data, bins)
        
        return X_data, y_data, y_binned, kf
        
        
        
    
    
    def _xgb_model(self):
    
        xgb_params = {'objective': 'reg:squarederror',
                      'random_state': 100,
                      'seed': 100}
        xgb_params.update(self.xgb_params)
        model = XGBRegressor(**xgb_params)
        
        return model
    
    
    
    def _lgbm_model(self):
    
        lgbm_params = {'objective':'regression', 'metric':'rmse'}
        lgbm_params.update(self.lgbm_params)
        model = LGBMRegressor(**lgbm_params)
        
        return model
    
    
    
    def _cat_model(self):
        
        categorical_cols = dtye_detection(self.df, 'category')
        cat_params = {'iterations': 1000,
              'task_type': 'GPU',
              'loss_function': 'RMSE',
              'cat_features' : categorical_cols,
              'grow_policy': 'Lossguide'
              }
        cat_params.update(self.cat_params)
        model = CatBoostRegressor(**cat_params)
        
        return model
    
    
    
    def _split_data(self, df, train_idx, test_idx, X_data, y_data):
        
        X_train, X_test = X_data.iloc[train_idx], X_data.iloc[test_idx]
        y_train, y_test = y_data.iloc[train_idx], y_data.iloc[test_idx]
        
        train_data = pd.concat((y_train, X_train),axis=1)
        train_data = filter_outlier(train_data)  
        y_train, X_train = train_data['price'], train_data.drop(('price'), axis=1)
        
        return X_train, X_test, y_train, y_test
        
        
    
    def _acc(self, truth, predict):
        
        rmse = np.sqrt(mean_squared_error(truth, predict))
        mae  = mean_absolute_error(truth, predict)
        r2 = r2_score(truth, predict)
        print(rmse)
        
        return pd.DataFrame({'RMSE':np.round(rmse,2),
                             'MAE':np.round(mae,2),
                             'R2-score':np.round(r2,2)}, index=['index_0',])
            
    
    
    def fit_model(self):
        
        X_data, y_data, y_binned, kf = self._split_X_y(self.df)
        xgb_X_data, xgb_y_data, xgb_y_binned, _ = self._split_X_y(self.xgb_df)
        
        predicts = []
        xgb_accs, lgbm_accs, cat_accs = [], [], []
        k=0
        for train_idx, test_idx in kf.split(X_data, y_binned):
                
            X_train, X_test, y_train, y_test = self._split_data(self.df, train_idx, test_idx, X_data, y_data)
            xgb_X_train, xgb_X_test, xgb_y_train, xgb_y_test = self._split_data(self.xgb_df, train_idx, test_idx, xgb_X_data, xgb_y_data)
            
            X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, train_size=0.5, test_size=0.5, random_state=100)
            xgb_X_test, xgb_X_val, xgb_y_test, xgb_y_val = train_test_split(xgb_X_test, xgb_y_test, train_size=0.5, test_size=0.5, random_state=100)
            
            xgb_model  = self._xgb_model()
            lgbm_model = self._lgbm_model()
            cat_model  = self._cat_model()
            
            xgb_model.fit(X_train, y_train,
                      eval_set=[(X_train, y_train), (X_val, y_val)],
                      early_stopping_rounds=10)
            
            lgbm_model.fit(X_train, y_train,
                           eval_set=(X_val, y_val),
                           early_stopping_rounds=10)
            
            cat_model.fit(X_train, y_train,
                           eval_set=(X_val, y_val),
                           early_stopping_rounds=10)
        
            xgb_result = xgb_model.predict(X_test, ntree_limit=0)
            lgbm_result= lgbm_model.predict(X_test)
            cat_result = cat_model.predict(X_test)
                        
            y_test, xgb_result = np.round(y_test, 4).astype(np.float64), np.round(xgb_result, 4).astype(np.float64)
            lgbm_result, cat_result = np.round(lgbm_result, 4).astype(np.float64), np.round(cat_result, 4).astype(np.float64)
            
            xgb_result, lgbm_result, cat_result = np.exp(xgb_result), np.exp(lgbm_result), np.exp(cat_result)
            y_test = np.exp(y_test)
            
            xgb_accs.append(self._acc(y_test, xgb_result))
            lgbm_accs.append(self._acc(y_test, lgbm_result))
            cat_accs.append(self._acc(y_test, cat_result))
            
            predict = pd.DataFrame({'truth': y_test, 'xgb': xgb_result, 'lgbm': lgbm_result, 'cat': cat_result})
            predicts.append(predict)
            

            with open('../models/xgboost/{}-fold.pickle'.format(k+1), mode='wb') as f:
                pickle.dump(xgb_model, f)
            with open('../models/lightgbm/{}-fold.pickle'.format(k+1), mode='wb') as f:
                pickle.dump(lgbm_model, f)
            with open('../models/catboost/{}-fold.pickle'.format(k+1) , mode='wb') as f:
                pickle.dump(cat_model, f)
            k+=1
        return xgb_accs, lgbm_accs, cat_accs, predicts
            
            
