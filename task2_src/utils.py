import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np



def correlation(df, figsize=(20, 17)):

    corr = df.corr()
    top_corr = corr.index[abs(corr['price'] > 0.5)]
    
    f, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=True)
    plt.tight_layout()
    ax.set_ylim(corr.shape[0], 0)
    plt.title('Correlation map')
    plt.savefig('./correlation_map.png')
    plt.close()
    
    f, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df[top_corr].corr(), annot=True, vmin=0.0, vmax=1.0)
    plt.tight_layout()
    ax.set_ylim(top_corr.shape[0], 0)
    plt.title('Correlation map over 0.5')
    plt.savefig('./over_0.5_correlation_map.png')
    plt.close()
    
    return top_corr



def detect_type(df, wanted_type='int'):
    
    for col in df.columns:
        
        if not df[col].types == wanted_type:
            df[col] = df[col].dtype(wanted_type)



def vis_boxplot(df, save_path, figsize=(12,10), vmax=None):
    
    f, ax = plt.subplots(figsize=figsize)
    plt.ylim(0, vmax)
    sns.boxplot(data=df)
    plt.savefig(save_path + '.png')
    
    
    
def vis_distplot(df, save_path, figsize=(12,10)):
    f, ax = plt.subplots(figsize=figsize)
    sns.distplot(df)
    plt.tight_layout()
    plt.savefig(save_path + '.png')
    
    
    
def filter_outlier(df, a=1.5):
    
    q= pd.DataFrame({'Q1': df.quantile(0.25), 'Q3': df.quantile(0.75)})
    iqr = pd.DataFrame({'IQR': q['Q3'] - q['Q1']})
    threshold = pd.DataFrame({'low': q['Q1'] - a*iqr['IQR'], 'high': q['Q3'] + a*iqr['IQR']})
    
    for col in df.columns:
        
        if df[col].dtype.name == 'category':
            pass
        elif threshold.loc[col, 'low'] == threshold.loc[col, 'high']:
            pass
        else:
            df = df.loc[(df[col] > threshold.loc[col, 'low']) & 
                        (df[col] < threshold.loc[col, 'high'])]
        
    return df
    


def dtye_detection(df, dtyep='category'):
    '''return columns'''    
    results = []
    for col in df.columns:
        
        if df[col].dtype.name == dtyep:
            results.append(col)
    
    return results
    

    