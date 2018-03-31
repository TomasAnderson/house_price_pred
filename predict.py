import hdb_train
import xgboost as xgb
import private_train

def hdb_predict(retrain=True):
    if retrain:
        X, y = hdb_train.load_data("./data/hdb_train.csv")
        model = xgb.XGBRegressor()
        model.fit(X, y)
        print(model)

        x_test = hdb_train.load_data("./data/hdb_test.csv", has_y=False)
        pred = model.predict(x_test)
        return pred


def private_predict(retrain=True):
    if retrain:
        X, y = private_train.load_data("./data/private_train.csv")
        model = xgb.XGBRegressor()
        model.fit(X, y)
        print(model)

        x_test = private_train.load_data("./data/private_test.csv", has_y=False)
        pred = model.predict(x_test)
        return pred

if __name__ == '__main__':
    with open("submission.csv", 'w') as output:
        output.write("index,price\n")
        pred = hdb_predict()
        for i, p in enumerate(pred):
            output.write("%d,%f\n"%(i, p))

        b_idx = 3766
        pred = private_predict()
        for i, p in enumerate(pred):
            idx = i + b_idx
            output.write("%d,%f\n"%(idx, p))
