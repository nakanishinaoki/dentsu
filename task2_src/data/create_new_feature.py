import pandas as pd
import numpy as np


def _des2cat(df, columns):
    '''Change to categorical type from descrete'''
    for column in columns:
        
        if column in df.columns:
            df[column] = df[column].astype('category')
    
    return df


def _deviation(sr, sr_ave):
    deviation = (sr - sr_ave) / sr.std() * 10 + 50
    return np.round(deviation, 3)



def _renovation_category(df):
    

    df['renovation_class'] = np.round(df['renovation_class']/10) + 1
    df.loc[df['renovation_class']<0, 'renovation_class'] = 0
    df['renovation_class'] = df['renovation_class'].astype(np.int64)
    df['renovation_class'] = df['renovation_class'].astype(str).str.zfill(3)
    
    return df



def _create_feature(df):
    
    df['deviation_sqft_living'] = _deviation(df['sqft_living'], df['sqft_living15'])
    df['deviation_sqft_lot'] =  _deviation(df['sqft_lot'], df['sqft_lot15'])
    
    df['renovation_class'] = df['yr_renovated'] - df['yr_built']
    df['renovation_class'] = np.round(df['renovation_class']/10) + 1
    df.loc[df['renovation_class']<0, 'renovation_class'] = 0
    df['renovation_class'] = df['renovation_class'].astype(np.int64)
    df['renovation_class'] = df['renovation_class'].astype(str).str.zfill(3)

    return df
    
    
    
    
    
if __name__ == '__main__':
        
    ori_df = pd.read_csv('../task_2/Price.csv')
    ori_df = _create_feature(ori_df)
    
    
    to_category_cols = list(('waterfront', 'view', 'condition', 'grade', 'renovation_class'))
    ori_df = _des2cat(ori_df, to_category_cols)
    
    ori_df.to_csv('Price_add_features.csv', index=False)
    
    
    
    
    
    