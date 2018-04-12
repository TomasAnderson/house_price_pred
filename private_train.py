import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def encode_category(col, n, one_hot=True):
    le = LabelEncoder()
    enc_col = le.fit_transform(col)
    enc_col.reshape(len(col), 1)
    if one_hot:
        enc = OneHotEncoder(sparse=False, n_values=n)
        enc_col = np.array(enc_col).reshape(-1, 1)
        enc_col = enc.fit_transform(enc_col)

    enc_col = np.array(enc_col)
    print("shape: ", enc_col.shape)
    return enc_col

def encode_month(col):
    year_col, mon_col = [], []
    for c in col:
        year, mon = c.split("-")
        year_col.append(float(year))
        mon_col.append(float(mon))
    return encode_num(year_col), encode_num(mon_col)


def encode_num(col):
    return np.nan_to_num(np.array(col)).reshape((len(col), 1))

def encode_year(col):
    N = len(col)
    year_col = np.zeros(N)
    for i, c in enumerate(col):
        year_col[i] = int(c[-4:])
    return year_col.reshape(N, 1)

def encode_binary(col, w):
    N = len(col)
    binary_col = np.zeros(N)
    for i, c in enumerate(col):
        if w in c:
            binary_col[i] = 1
        else:
            binary_col[i] = 0
    return binary_col.reshape((N,1))

def encode_completion_date(col, n):
    categories = ['Uncompleted','Uncomplete','Unknown']
    N = len(col)
    date_col = np.zeros(N)
    for i, c in enumerate(col):
        if c not in categories:
            # TODO: add range encoding here
            date_col[i] = 3
        else:
            date_col[i] = categories.index(c)
    enc = OneHotEncoder(sparse=False, n_values=n)
    enc_col = np.array(date_col).reshape(-1, 1)
    enc_col = enc.fit_transform(enc_col)
    print("shape: ", enc_col.shape)
    return enc_col


def load_data(f, has_y=True):
    df = pd.read_csv(f)

    # categorical feature
    land_type_col = encode_category(df['type_of_land'], n=3)
    property_col = encode_category(df['property_type'], n=6)
    completion_date_col = encode_completion_date(df['completion_date'], n=4)
    sale_type_col = encode_category(df['type_of_sale'], n=3)
    # tenure_col = encode_binary(df['tenure'], w='Freehold')     # is tenure
    postal_district_col = encode_category(df['postal_district'], n=27)
    region_col = encode_category(df['region'], n=5)
    area_col = encode_category(df['area'], n=40)

    # date feature
    year_col = encode_year(df['contract_date'])
    sale_year_col, _ = encode_month(df['month'])

    # numeric feature
    postal_col = encode_num(df['postal_code'])
    lat_col = encode_num(df['latitude'])
    lon_col = encode_num(df['longitude'])
    floor_col = encode_num(df['floor_num'])


    # block_col = encode_category(df['project_name']) # dim = 3523
    # street_col = encode_category(df['street_name']) # dim =311913
    tenure_col = encode_category(df['tenure'], n=850) # dim = 850
    # unit_col

    encoded_x = None

    for i, col in enumerate([land_type_col, property_col, completion_date_col, sale_type_col, tenure_col, postal_district_col, region_col, area_col,
                             year_col, sale_year_col, postal_col, lat_col, lon_col, floor_col]):
        if encoded_x is None:
            encoded_x = col
        else:
            encoded_x = np.concatenate((encoded_x, col), axis=1)


    if has_y:
        y = np.array(df['price'])
        print("X train shape: : ", encoded_x.shape)
        print("y train shape: : ", y.shape)
        return encoded_x, y
    else:
        print("X test shape: : ", encoded_x.shape)
        return encoded_x


if __name__ == '__main__':
    X, y = load_data("./data/private_train.csv")

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
