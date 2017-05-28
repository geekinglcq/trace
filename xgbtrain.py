import xgboost as xgb
import numpy as np

def train(datafile, modelfile):
    dtrain = xgb.DMatrix(datafile)
    param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic'}
    num_round = 20
    print ('running cross validation')
    bst = xgb.train(param, dtrain, num_round, [(dtrain, 'eval'),(dtrain, 'train')])
    bst.save_model(modelfile)
    print('Model has been saved as %s\n'%(modelfile))
    return bst




def predict(modelfile, datafile):
    bst = xgb.Booster(model_file=modelfile)
    test = xgb.DMatrix(datafile)
    pred = bst.predict(datafile)