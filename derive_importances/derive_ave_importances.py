import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, Lasso, LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import train_test_split



class Models():
    
    def __init__(self, file):
        
        df = pd.read_csv(file)
        df.drop((['id', 'date']), axis=1, inplace=True)
        self.category_cols = list(('waterfront', 'view', 'condition', 'grade', 'renovation_class'))
        
        self.df = _des2cat(df, self.category_cols)
    
        X_data = df.drop(('price'), axis=1)
        y_data = np.log(df['price'])
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_data, y_data, train_size=0.8, test_size=0.2, random_state=100)
        
        
        
    def random_forest(self):
        
        print('Random_Forest')
        
        rg = RFR(n_jobs=-1, n_estimators=100, random_state=100)
        rg.fit(self.X_train, self.y_train)
        
        importances = pd.DataFrame({'RF': rg.feature_importances_}, index=self.X_train.columns)
        importances = _norm(importances)
        
        return rg, importances
    
    
    
    def lasso(self):
        
        print('LASSO')
        clf = Lasso(alpha=0.1)
        clf.fit(self.X_train, self.y_train)
        
        pred = clf.predict(self.X_test)
        importances = pd.DataFrame(({'LASSO': np.abs(clf.coef_)}), index=self.X_train.columns)
        importances = _norm(importances)
        
        
        self.y_test, pred = np.exp(self.y_test), np.exp(pred)
        
        return clf, importances
    
    
    
    def linear_regression(self):
        '''Recursive Feature Elimination (RFE)'''
        print('LinearRegression')
        lr = LinearRegression(normalize=True)
        lr.fit(self.X_train, self.y_train)
        
        linreg = pd.DataFrame(({'LinReg': np.abs(lr.coef_)}), index=self.X_train.columns)
        linreg = _norm(linreg)
        
        rfe = RFE(lr, n_features_to_select=1, verbose=1)
        rfe.fit(self.X_train, self.y_train)
        
        RFE_ = pd.DataFrame(({'RFE': rfe.ranking_}), index=self.X_train.columns)
        RFE_ = _norm(RFE_)
        
        ridge = Ridge(alpha=0.7)
        ridge.fit(self.X_train, self.y_train)
        ridge = pd.DataFrame(({'Ridge': np.abs(ridge.coef_)}), index=self.X_train.columns)
        ridge = _norm(ridge)
        
        return RFE_, ridge, linreg
    
    
    
    def xgboost(self):
        
        print('XGBoost')
        model = XGBRegressor()
        X_train = _change_dtype(self.X_train, ['category'], 'int32')
        model.fit(X_train, self.y_train)
        self.XGB_ = pd.DataFrame(({'XGB': model.feature_importances_}), index=X_train.columns)
        
        
        XGB_ = pd.DataFrame(({'XGB': model.feature_importances_}), index=X_train.columns)
        XGB_ = _norm(XGB_)
        
        return XGB_
    
    
    
    def lightGBM(self):
        
        print('LightGBM')
        model = LGBMRegressor()
        model.fit(self.X_train, self.y_train)
        
        lgb = pd.DataFrame(({'LightGBM': model.feature_importances_}), index=self.X_train.columns)
        lgb = _norm(lgb)
        
        return lgb
    
    
    
    def catbooost(self):
        
        print('CatBoost')
        model = CatBoostRegressor(cat_features = self.category_cols)
        model.fit(self.X_train, self.y_train)
        
        cat = pd.DataFrame(({'CatBoost': model.feature_importances_}), index=self.X_train.columns)
        cat = _norm(cat)
        
        return cat




def _norm(df):
    
    minmax = MinMaxScaler()
    
    columns, index = df.columns, df.index
    out = minmax.fit_transform(np.array(df).astype(np.float64))
    out = np.round(out, 2)
    
    return pd.DataFrame(out, columns=columns, index=index)



def _des2cat(df, columns):
    '''Change to categorical type from descrete'''
    for column in columns:
        
        if column in df.columns:
            df[column] = df[column].astype('category')
    
    return df



def _change_dtype(df, search=[], change='int32'):
    '''
    return columns
    search must be list
    cahnge must be str
    '''
    for col in df.columns:
        
        if df[col].dtype.name in search:
            df[col] = df[col].astype(change)
    
    return df


def heatmap(df, save_path, figsize=(20,15)):
    
    f, ax = plt.subplots(figsize=figsize)
    sns.set(font_scale=1.5)
    sns.set_context('talk')
    sns.heatmap(data=df, annot=True)
    ax.set_ylim(df.shape[0], 0)
    plt.tight_layout()
    plt.title('importances', fontsize=20)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()    




#%%
if __name__ == '__main__':
    
    
    models = Models(file='../Price_add_features.csv')
    
    rfr_model, rfr_importances = models.random_forest()
    
    lasso_model, lasso_importance = models.lasso()
    
    RFE_usinflr_importances, ridge_importances, linear_regression_importances = models.linear_regression()
    
    XGB_importances = models.xgboost()
    
    lgb_importances = models.lightGBM()
    
    catboost_importances = models.catbooost()
    
    importances = pd.concat((rfr_importances,
                             lasso_importance,
                             RFE_usinflr_importances,
                             ridge_importances,
                             linear_regression_importances,
                             XGB_importances,
                             lgb_importances,
                             catboost_importances), axis=1, sort=False)
    
    mean = pd.DataFrame({'mean': np.round(importances.mean(axis=1),2)})
    importances = pd.concat((mean, importances), axis=1).sort_values(by='mean', ascending=False)
    
    
    heatmap(importances, save_path='./importances.png')
    importances.to_csv('importances.csv', index=False)
    

    
    
    
    
    
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



