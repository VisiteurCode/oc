import joblib
import pandas as pd
import sklearn

def predict(data, full_data):
    lr_pipeline = joblib.load('./logreg_pipeline.joblib')
    result = lr_pipeline.predict(data)
    y_pred = pd.DataFrame({'Is_Genuine': result})
    pred = pd.concat([y_pred, full_data], axis=1)
    return pred
