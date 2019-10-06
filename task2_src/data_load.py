import pandas as pd
import numpy as np
from utils import correlation, vis_boxplot
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression



class LoadData():
    
    def __init__(self, file, std=False, norm=False, all_data=False, corr_data=False,
                 kbest=False, ave_importances=False, target='price'):
        self.df = pd.read_csv(file)
        self.df.drop(['id', 'date'], axis=1, inplace=True)
        self.std = std
        self.norm = norm
        self.all_data_bool = all_data
        self.corr_data_bool = corr_data
        self.kbest_bool = kbest
        self.ave_importances = ave_importances
        self.target = target
        self.category_cols = list(('waterfront', 'view', 'condition', 'grade', 'renovation_class'))
        self.dtypes = ['int64', 'int32', 'int16', 'float64', 'float32', 'float16']
    

        
    def _quantitarive(self, df):
        quantitative = [f for f in df.columns if df[f].dtypes.name in self.dtypes]
        return quantitative



    def _qualitative(self, df):
        qualitative = [f for f in df.columns if not df[f].dtypes.name in self.dtypes]
        return qualitative                
            
        

    def _normalization(self, df):
        columns = self._quantitarive(df)
        for col in columns:
            target = df[col].copy()
            t_max, t_min = np.max(target), np.min(target)
            target = (target - t_min)/(t_max - t_min)
            df.loc[:, col] = target
            
        return df



    def _standardization(self, df):
        columns = self._quantitarive(df)
        for col in columns:
            target = df[col].copy()
            t_max, t_min = np.max(target), np.min(target)
            target = (target - t_min)/(t_max - t_min)
            df.loc[:, col] = target
            
        return df
    
    
    
    def _std_norm(self, df):
        '''X_data consist of excluded target values'''
        
        X_data = df.drop(('price'), axis=1)
        y_data = pd.DataFrame(df['price'])        
        
        if self.std==True and self.norm==True:
            raise ValueError('NOT approval both True')
        elif self.std:
            X_data = self._standardization(X_data)
        elif self.norm:
            X_data = self._normalization(X_data)
        
        return pd.concat((y_data,X_data),axis=1)
    
    
    
    def _des2cat(self, df, columns):
        '''Change to categorical type from descrete'''
        for column in columns:
            
            if column in df.columns:
                df[column] = df[column].astype('category')
        
        return df
    
    
    
    def _filter_outlier_select_oneself(self, df):
        
        min_max = {}
        min_max.update({'price': {'min': np.log(7500), 'max': np.log(4668000)}})
        min_max.update({'sqft_living': {'min': 200, 'max': 10040}})
        min_max.update({'sqft_lot15': {'min': 650, 'max': 560617}})
        min_max.update({'sqft_lot': {'min': 500, 'max': 432036}})        
        
        for col in  ['price', 'sqft_living', 'sqft_lot15', 'sqft_lot']:
            
            if col in df.columns:

                df = df.loc[(df[col] > min_max[col]['min']) &
                            (df[col] < min_max[col]['max'])]
        
        return df
    
    
    
    def _filter_outlier(self, df, categorical_cols=None, a=1.5):
        
        
        q= pd.DataFrame({'Q1': df.quantile(0.25), 'Q3': df.quantile(0.75)})
        iqr = pd.DataFrame({'IQR': q['Q3'] - q['Q1']})
        threshold = pd.DataFrame({'low': q['Q1'] - a*iqr['IQR'], 'high': q['Q3'] + a*iqr['IQR']})
        
        
        if categorical_cols == None:
            cols = df.columns
        else:
            cols = df.drop(categorical_cols, axis=1).columns
        
        for col in cols:
            if threshold.loc[col, 'low'] == threshold.loc[col, 'high']:
                pass
            else:
                df = df.loc[(df[col] > threshold.loc[col, 'low']) & 
                            (df[col] < threshold.loc[col, 'high'])]
            
        return df
            
    
    
    
    def _kbest(self, df, n_k=5, n_feature=10):
        
        predictor = df.drop(('price'), axis=1).columns
        selection = SelectKBest(f_regression, k=n_k).fit(df.drop(('price'),axis=1), df['price'])
        scores = selection.scores_
        p_val  = selection.pvalues_
        
        scores_df = pd.DataFrame({'feature_importance': scores}, index=predictor)
        p_val = pd.DataFrame({'p-val': p_val}, index=predictor)
        
        selected_feature = pd.concat((scores_df, p_val), axis=1).sort_values('feature_importance', ascending=False)
        # write
        selected_feature.to_csv('feature_importance_using_kbest.csv')
        selected_feature = selected_feature.iloc[:n_feature, :]
        
        return selected_feature
    
    
    
    def selected_data_using_average_importances(self):
        '''
        average feature importances
        using methods which are RandomForest, LASSO, LinearRegression, RFE(LinReg),
        Ridge(LinReg), XGB, LightGBM and CatBoost
        '''
        
        features = list(('grade', 'lat', 'sqft_living', 'deviation_sqft_lot', 'yr_built', 
                        'deviation_sqft_living', 'zipcode','long', 'sqft_lot', 'sqft_lot15',
                        'renovation_class', 'yr_renovated', 'sqft_living15', 'sqft_above', 'view', 'condition'))

        X_data = self.df.loc[:, features]
        X_data = self._des2cat(X_data, self.category_cols)
        y_data = np.log(self.df['price'])
        
        # vis X_data
        self.max_X_data = np.max(X_data.max())
        vis_boxplot(X_data, save_path='./selected_data_using_ave_FeatureImportances', vmax=self.max_X_data)
        
        y_data = pd.DataFrame(y_data)
        
        return pd.concat((y_data, X_data), axis=1)
        
    
    
    def selected_data_using_kbest(self):
        
        features = self._kbest(self.df, n_feature=11)
        X_data = self.df.ix[:, features.index]
        X_data = self._des2cat(X_data, self.category_cols)
        y_data = np.log(self.df['price'])

        # vis X_data
        self.max_X_data = np.max(X_data.max())
        vis_boxplot(X_data, save_path='./selected_data_using_kboost', vmax=self.max_X_data)
        
        y_data = pd.DataFrame(y_data)
        
        return pd.concat((y_data, X_data), axis=1)
    
        
        
        
    def selected_data_using_corr(self):
        # selected data using corr.
    
        df = self._des2cat(self.df, self.category_cols)
        
        X_data = df.drop(['price'], axis=1)
        y_data = np.log(df['price'])
        
        top_corr = correlation(df)
        X_data = X_data.loc[:, top_corr[1:]]

        # vis X_data
        self.max_X_data = np.max(X_data.max())
        vis_boxplot(X_data, save_path='./selected_data_using_corr', vmax=self.max_X_data)
        
        y_data = pd.DataFrame(y_data)
        
        return pd.concat((y_data, X_data), axis=1)
    
    
    
    def all_data(self):
    
        df = self._des2cat(self.df, self.category_cols)
        
        df['price'] = np.log(df['price'])
        df = self._filter_outlier(df, self.category_cols)
        df = self._std_norm(df)
        
        # vis X_data
        self.max_X_data = np.max(df.drop(('price'), axis=1).max())
        vis_boxplot(df.drop(['price'], axis=1), save_path='./selected_data', vmax=self.max_X_data)
        
        return df
    
    
    
    def divid_data(self):
        
        if np.sum((self.all_data_bool, self.corr_data_bool, self.kbest_bool)) > 1:
            raise ValueError('MUST be only one True or all False')
        elif self.all_data_bool:
            df = self.all_data()
        elif self.corr_data_bool:
            df = self.selected_data_using_corr()
        elif self.kbest_bool:
            df = self.selected_data_using_kbest()
        elif self.ave_importances:
            df = self.selected_data_using_average_importances()
        else:
            df = self.df.copy()
        
        X_data, y_data = df.drop(('price'), axis=1), df['price']
        bins = np.linspace(int(y_data.min()), int(y_data.max())+1, 10)
        y_binned = np.digitize(y_data, bins)
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data,
                                                            train_size=0.8, test_size=0.2,
                                                            random_state=100, stratify=y_binned)

        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test,
                                                        train_size=0.5, test_size=0.5,
                                                        random_state=100)
        
        dtrain = pd.concat((y_train, X_train), axis=1)
        dtest = pd.concat((y_test, X_test), axis=1)
        dval = pd.concat((y_val, X_val), axis=1)
        
        categorical_cols = set(self.category_cols) & set(list(X_train.columns))
        dtrain = self._filter_outlier(dtrain, categorical_cols)
        vis_boxplot(dtrain.drop(('price'), axis=1), save_path='./removed_outlier_data', vmax=self.max_X_data)
        
        dtrain = self._std_norm(dtrain)
        dtest = self._std_norm(dtest)
        dval = self._std_norm(dval)
        
        return dtrain, dtest, dval

    
        


    