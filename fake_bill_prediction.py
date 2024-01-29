import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
import numpy as np

def predict(data):
    lr_pipeline = joblib.load('./logreg_pipeline.joblib')
    result = lr_pipeline.predict(data)
    y_pred = pd.DataFrame({'is_genuine': result})
    pred = pd.concat([y_pred, data], axis=1)
    return pred
