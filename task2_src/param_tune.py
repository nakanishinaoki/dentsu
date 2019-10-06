import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import optuna
from utils import dtye_detection

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor



class Param_tuning():
    
    def __init__(self, dtrain):
        
        self.X_data, self.y_data = dtrain.drop(('price'), axis=1), dtrain['price']

        bins = np.linspace(int(self.y_data.min()), int(self.y_data.max())+1, 10)
        self.y_binned = np.digitize(self.y_data, bins)
        
        
        
    def _split_data(self, X_data):
        
        X_train, X_test, y_train, y_test = train_test_split(X_data, self.y_data, 
                                                            train_size=0.8, test_size=0.2,
                                                            random_state=100, stratify=self.y_binned)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test,
                                                        train_size=0.5, test_size=0.5,
                                                        random_state=100)
        
        return X_train, X_test, X_val, y_train, y_test, y_val
        
        
        

    def _lgb_opt(self, trial):
        
        # Classifier
        X_train, X_test, X_val, y_train, y_test, y_val = self._split_data(self.X_data)
        
        # paramter_tuning using optuna
        bagging_freq =  trial.suggest_int('bagging_freq',1,10),
        min_data_in_leaf =  trial.suggest_int('min_data_in_leaf',2,100),
        max_depth = trial.suggest_int('max_depth',1,20),
        learning_rate = trial.suggest_loguniform('learning_rate',0.001,0.1),
        num_leaves = trial.suggest_int('num_leaves',2,70),
        num_threads = trial.suggest_int('num_threads',1,10),
        min_sum_hessian_in_leaf = trial.suggest_int('min_sum_hessian_in_leaf',1,10),
        boosting = trial.suggest_categorical('boosting', ['gbdt', 'dart', 'goss'])
        
        
        lightgbm_tuna = LGBMRegressor(
                random_state = 100,
                verbosity = 10,
                bagging_seed = 100,
                boost = boosting,
                bagging_freq = bagging_freq ,
                min_data_in_leaf = min_data_in_leaf,
                max_depth = max_depth,
                learning_rate = learning_rate,
                num_leaves = num_leaves,
                num_threads = num_threads,
                min_sum_hessian_in_leaf = min_sum_hessian_in_leaf,
                objective = 'regression',
                metric = 'rmse',
                )
        
        lightgbm_tuna.fit(X_train, y_train)
        pred = lightgbm_tuna.predict(X_test)
        
        y_test, pred = np.round(y_test, 4), np.round(pred, 4)
        y_test, pred = np.exp(y_test), np.exp(pred)
        
        return np.sqrt(mean_squared_error(pred, y_test))
    
    
    
    def _xgb_opt(self, trial):
        
        X_data = pd.get_dummies(self.X_data)
        # Classifier
        X_train, X_test, X_val, y_train, y_test, y_val = self._split_data(X_data)
        
        #paramter_tuning using optuna
        max_depth = trial.suggest_int('max_depth',1,20)
        learning_rate = trial.suggest_loguniform('learning_rate',0.001, 0.1)
        subsample = trial.suggest_discrete_uniform('subsample', 0.5, 0.9, 0.1)
        booster = trial.suggest_categorical('booster', ['gbtree', 'gblinear'])
        colsample_bytree = trial.suggest_discrete_uniform('colsample_bytree', 0.5, 0.9, 0.1)
        eval_metric = trial.suggest_categorical('eveal_metric', ['mae', 'rmse'])
        
        XGboost_tuna = XGBRegressor(random_state = 100,
                                    verbosity = 1,
                                    bagging_seed = 100,
                                    max_depth = max_depth,
                                    learning_rate = learning_rate,
                                    subsample = subsample,
                                    booster = booster,
                                    colsample_bytree = colsample_bytree,
                                    objective = 'reg:squarederror',
                                    eval_metric = eval_metric)
        
        XGboost_tuna.fit(X_train, y_train)
        pred = XGboost_tuna.predict(X_test)
        
        y_test, pred = np.round(y_test, 4), np.round(pred, 4)
        y_test, pred = np.exp(y_test), np.exp(pred)
    
        return np.sqrt(mean_squared_error(pred, y_test))
    
    
    
    def _cat_opt(self, trial):
    
        # Classifier
        X_train, X_test, X_val, y_train, y_test, y_val = self._split_data(self.X_data)
        
        # paramter_tuning using optuna
        cat_features = dtye_detection(X_train, 'category')
        depth = trial.suggest_int('depth', 1, 20)
        bagging_temperature = trial.suggest_discrete_uniform('bagging_temperature', 1, 21, 5)
        learning_rate = trial.suggest_loguniform('learning_rate',0.001,0.1)
        num_leaves = trial.suggest_int('num_leaves', 2, 70)
        min_data_in_leaf =  trial.suggest_int('min_data_in_leaf', 2, 100)
        thread_count = trial.suggest_int('thread_count', 1, 10)
        
        catboost_tuna = CatBoostRegressor(
                random_state = 0,
                verbose = 100,
                loss_function = 'RMSE',
                task_type = 'GPU',
                depth = depth,
                bagging_temperature = bagging_temperature,
                learning_rate = learning_rate,
                num_leaves = num_leaves,
                min_data_in_leaf = min_data_in_leaf,
                thread_count = thread_count,
                cat_features = cat_features,
                grow_policy = 'Lossguide'
                )
        
        catboost_tuna.fit(X_train, y_train)
        pred = catboost_tuna.predict(X_test)
        
        y_test = np.exp(y_test)
        pred = np.exp(pred)
        
        return np.sqrt(mean_squared_error(pred, y_test))
        
    
    
    def param_tune(self, n_traials=150):
        
        xgb_study = optuna.create_study()
        xgb_study.optimize(self._xgb_opt, n_trials=n_traials)
        print('xgb_params', xgb_study.best_params)
        
        lgb_study = optuna.create_study()
        lgb_study.optimize(self._lgb_opt, n_trials=n_traials)
        print('lgb_params', lgb_study.best_params)
        
        cat_study = optuna.create_study()
        cat_study.optimize(self._cat_opt, n_trials=n_traials)
        print('cat_params', cat_study.best_params)
    
        return xgb_study.best_params, lgb_study.best_params, cat_study.best_params
    
    
    
    
    



