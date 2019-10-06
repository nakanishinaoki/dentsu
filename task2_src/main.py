import os
import pandas as pd
import numpy as np

from param_tune import Param_tuning
from data_load import LoadData
from models import Models
from stacking import Stack_models
import argparse


if __name__ == '__main__':
    ''' Choice using data
        -> alldata : all_data=True
        -> over 0.5 correlation data : corr_data=True
        -> selected kbest data : kbest=True
        -> ave_importances data : ave_importances=True
    '''
    parser = argparse.ArgumentParser(description='house price regression')
    parser.add_argument('--all_data', type=bool, default=False)
    parser.add_argument('--corr_data', type=bool, default=False)
    parser.add_argument('--kbest', type=bool, default=False)
    parser.add_argument('--ave_importances', type=bool, default=True)
    parser.add_argument('--n_traials', type=int, default=1)
    
    args = parser.parse_args()

    
    os.makedirs('../models/xgboost', exist_ok=True)
    os.makedirs('../models/lightgbm', exist_ok=True)
    os.makedirs('../models/catboost', exist_ok=True)
    
    file='./data/Price_add_features.csv'
    loder = LoadData(file=file, std=True, 
                     all_data=args.all_data, corr_data=args.corr_data, 
                     kbest=args.kbest, ave_importances=args.ave_importances)
    dtrain, dtest, dval = loder.divid_data()

    param_tune = Param_tuning(dtrain)
    xgb_params, lgb_params, cat_params = param_tune.param_tune(n_traials=args.n_traials)

    df = pd.read_csv(file)
    df.drop((['id', 'date']), axis=1, inplace=True)
    df['price'] = np.log(df['price'])
    
    models = Models(df, xgb_params, lgb_params, cat_params)
    xgb_accs, lgbm_accs, cat_accs, predictions = models.fit_model()
    
    stack_model = Stack_models(predictions)
    stac_accs, stac_results = stack_model.stacking()

    
    
    
    if not os.path.isdir('../results'):
        os.mkdir('../results')
    
    rmse, mae, r2 = 0, 0, 0
    for k in range(len(stac_accs)):
        rmse += stac_accs[k]['RMSE']
        mae  += stac_accs[k]['MAE']
        r2 += stac_accs[k]['R2-score']
    
    with open('../results/result.txt', 'a') as f:
            f.write('RMSE: ' + str(rmse / len(stac_accs)) + '\n'+ \
                    'MAE: ' + str(mae / len(stac_accs)) + '\n' + \
                    'R2-score: ' + str(r2 / len(stac_accs))+ '\n\n')    
        


    for k in range(len(stac_accs)):
        
        with open('../results/stacking_result.txt', 'a') as f:
            f.write(str(k) + '-fold stacking_model' + '\n'+\
                    'RMSE: ' + str(stac_accs[k]['RMSE']) + '\n'+ \
                    'MAE: ' + str(stac_accs[k]['MAE']) + '\n' + \
                    'R2-score: ' + str(stac_accs[k]['R2-score'])+ '\n\n')
        
        with open('../results/xgboost_result.txt', 'a') as f:
            f.write(str(k) + '-fold stacking_model' + '\n'+\
                    'RMSE: ' + str(xgb_accs[k]['RMSE']) + '\n'+ \
                    'MAE: ' + str(xgb_accs[k]['MAE']) + '\n' + \
                    'R2-score: ' + str(xgb_accs[k]['R2-score'])+ '\n\n')
        
        with open('../results/lgbm_result.txt', 'a') as f:
            f.write(str(k) + '-fold stacking_model' + '\n'+\
                    'RMSE: ' + str(lgbm_accs[k]['RMSE']) + '\n'+ \
                    'MAE: ' + str(lgbm_accs[k]['MAE']) + '\n' + \
                    'R2-score: ' + str(lgbm_accs[k]['R2-score'])+ '\n\n')
            
        with open('../results/cat_coost_result.txt', 'a') as f:
            f.write(str(k) + '-fold stacking_model' + '\n'+\
                    'RMSE: ' + str(cat_accs[k]['RMSE']) + '\n'+ \
                    'MAE: ' + str(cat_accs[k]['MAE']) + '\n' + \
                    'R2-score: ' + str(cat_accs[k]['R2-score'])+ '\n\n')



    