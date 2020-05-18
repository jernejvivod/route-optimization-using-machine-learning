import numpy as np
import xgboost as xgb

class XGBPredictor():
    def __init__(self, model):
        self.model = model

    def predict(self, data):
        return np.exp(self.model.predict(xgb.DMatrix(data))) - 1

