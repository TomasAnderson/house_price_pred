import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from sklearn.metrics import mean_squared_error

rng = np.random.RandomState(31337)


def encode_category(col, one_hot=True):
    le = LabelEncoder()
    enc_col = le.fit_transform(col)
    enc_col.reshape(len(col), 1)
    if one_hot:
        enc = OneHotEncoder(sparse=False)
        enc_col = np.array(enc_col).reshape(-1, 1)
        enc_col = enc.fit_transform(enc_col)
    return np.array(enc_col)

def encode_month(col):
    year_col, mon_col = [], []
    for c in col:
        year, mon = c.split("-")
        year_col.append(float(year))
        mon_col.append(float(mon))
    return year_col, mon_col


def load_data():
    df = pd.read_csv("./data/hdb_train.csv")

    features = ['flat_type', 'floor_area_sqm', 'month', 'street_name', 'postal_code', 'floor']

    type_col = encode_category(df['flat_type'])
    N = len(type_col)
    area_col = np.array(df['floor_area_sqm']).reshape((N, 1))
    year_col, _ = encode_month(df['month'])
    year_col = np.array(year_col).reshape((N,1))
    street_col = encode_category(df['street_name'])
    floor_col = np.array(df['floor']).reshape((N,1))
    encoded_x = None

    for i, col in enumerate([type_col, street_col, area_col, year_col, floor_col]):
        if encoded_x is None:
            encoded_x = col
        else:
            encoded_x = np.concatenate((encoded_x, col), axis=1)

    y = np.array(df['resale_price'])
    print("X shape: : ", encoded_x.shape)
    print("y shape: : ", y.shape)
    return encoded_x, y

X, y = load_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=31337)

model = xgb.XGBRegressor()
model.fit(X_train, y_train)
print(model)

# make predictions for test data
predictions = model.predict(X_test)


# evaluate predictions
mse = mean_squared_error(y_test, predictions)
print("MSE: %.2f" % (mse))
