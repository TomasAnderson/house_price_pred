import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def encode_category(col, one_hot=True):
    le = LabelEncoder()
    enc_col = le.fit_transform(col)
    enc_col.reshape(len(col), 1)
    if one_hot:
        enc = OneHotEncoder(sparse=False)
        enc_col = np.array(enc_col).reshape(-1, 1)
        enc_col = enc.fit_transform(enc_col)
    return np.array(enc_col)

def encode_model(col):
    models = ['DBSS', 'MULTI GENERATION', 'MODEL A2', 'PREMIUM MAISONETTE', 'SIMPLIFIED',
              'PREMIUM APARTMENT LOFT', 'TERRACE', 'STANDARD', 'IMPROVED', 'MAISONETTE',
              'MODEL A', 'PREMIUM APARTMENT', 'ADJOINED FLAT', 'IMPROVED-MAISONETTE', 'TYPE S1',
              'TYPE S2', '2-ROOM', 'MODEL A-MAISONETTE', 'APARTMENT', 'NEW GENERATION']

    le = LabelEncoder().fit(models)
    enc_col = le.transform(col)
    enc = OneHotEncoder(sparse=False, n_values=len(models))
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

def encode_range(col):
    range_col = np.zeros((len(col),2))
    for i, c in enumerate(col):
        m, n = c.split("TO")
        range_col[i] = [int(m), int(n)]
    return range_col

def encode_num(col):
    return np.array(col).reshape((len(col), 1))

def load_data(f, has_y=True):
    df = pd.read_csv(f)

    # categorical feature

    model_col = encode_model(df['flat_model'])
    type_col = encode_category(df['flat_type'])

    # range feature
    storey_col = encode_range(df['storey_range'])


    # numeric feature
    lease_commence_col = encode_num(df['lease_commence_date'])
    postal_col = encode_num(df['postal_code'])
    area_col = encode_num(df['floor_area_sqm'])
    floor_col = encode_num(df['floor'])
    lat_col = encode_num(df['latitude'])
    lon_col = encode_num(df['longitude'])
    year_col, _ = encode_month(df['month'])
    year_col = encode_num(year_col)


    # block_col = encode_category(df['block'])
    # street_col = encode_category(df['street_name'])

    encoded_x = None

    for i, col in enumerate([model_col, type_col, area_col, year_col, floor_col, lat_col, lon_col, storey_col, lease_commence_col, postal_col]):
        print(i, col.shape)
        if encoded_x is None:
            encoded_x = col
        else:
            encoded_x = np.concatenate((encoded_x, col), axis=1)


    if has_y:
        y = np.array(df['resale_price'])
        print("X train shape: : ", encoded_x.shape)
        print("y train shape: : ", y.shape)
        return encoded_x, y
    else:
        print("X test shape: : ", encoded_x.shape)
        return encoded_x

if __name__ == '__main__':
    X, y = load_data("./data/hdb_train.csv")
    rng = np.random.RandomState(31337)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=31337)

    # model = xgb.XGBRegressor()
    model = RandomForestRegressor()

    model.fit(X_train, y_train)
    print(model)

    # make predictions for test data
    predictions = model.predict(X_test)

    # evaluate predictions
    mse = mean_absolute_percentage_error(y_test, predictions)
    print("MAPE: %.2f" % (mse))
