import pandas as pd
import numpy as np
import os
import pickle
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression


class Stack_models():
    
    def __init__(self, df):
        
        self.df = df
        
    
    
    def stacking(self):
        
        accs, results = [], []
        
        for k in range(len(self.df)):
            X_predictions = self.df[k].drop(('truth'), axis=1)
            y_predictions = pd.DataFrame(self.df[k]['truth'])
            
            X_train, X_test, y_train, y_test = train_test_split(X_predictions, y_predictions, train_size=0.8, test_size=0.2, random_state=100)
            meta_model = LinearRegression()
            meta_model.fit(X_train, y_train)
            pred = pd.DataFrame({'pred': np.squeeze(meta_model.predict(X_test))}, index=X_test.index)
            
            results.append(pd.concat((y_test, pred), axis=1))
            
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            mae = mean_absolute_error(y_test, pred)
            r2 = r2_score(y_test, pred)
            acc = {'RMSE': rmse, 'MAE': mae, 'R2-score': r2}
            accs.append(acc)
            
            
            if not os.path.isdir('../models/stacking'):
                os.mkdir('../models/stacking')
        
            with open('../models/stacking/RMSE_{:.2f}_RAE{:.2f}_R2_{:.2f}.pickle'.format(rmse, mae, r2) , mode='wb') as f:
                pickle.dump(meta_model, f)
        
        return accs, results
            
