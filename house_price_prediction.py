# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
# train_data = pd.read_csv("/kaggle/input/home-data-for-ml-course/train.csv")
# test_data = pd.read_csv("/kaggle/input/home-data-for-ml-course/test.csv")
train_data = pd.read_csv(
    "./house-prices-advanced-regression-techniques/train.csv")
test_data = pd.read_csv(
    "./house-prices-advanced-regression-techniques/test.csv")

# with open('/kaggle/input/home-data-for-ml-course/data_description.txt') as f:
#     content = f.read()
#     print(content)

features = ["LotArea", "Street", "YearBuilt", "LandSlope", "HouseStyle", "BedroomAbvGr", "KitchenQual", "GarageArea", "GarageQual"]
# train_data = train_data[features]
# features = ["LotArea", "BedroomAbvGr", "GarageArea"]
# features = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
# train_data.dropna(inplace=True)
# drop_features = ['SalePrice', 'Id']
# X = train_data.drop(drop_features, axis=1, inplace=False)
X = train_data[features]
cat = X.select_dtypes(include="object")
num = X.select_dtypes(exclude="object")
y = train_data["SalePrice"]

cat_train, cat_val, num_train, num_val, y_train, y_val = train_test_split(
    cat, num, y, test_size=0.2, random_state=42)

# missing value
imputer_num = SimpleImputer(strategy="mean")
num_train_imputed = pd.DataFrame(imputer_num.fit_transform(num_train))
num_val_imputed = pd.DataFrame(imputer_num.transform(num_val))
num_train_imputed.columns = num_train.columns
num_val_imputed.columns = num_val.columns
print(f" num_train_imputed shape: {num_train_imputed.shape}")
imputer_cat = SimpleImputer(strategy="constant", fill_value="Unknown")
cat_train_imputed = pd.DataFrame(imputer_cat.fit_transform(cat_train))
cat_val_imputed = pd.DataFrame(imputer_cat.transform(cat_val))
cat_train_imputed.columns = cat_train.columns
cat_val_imputed.columns = cat_val.columns
print(f" cat_train_imputed shape: {cat_train_imputed.shape}")

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
cat_data = encoder.fit_transform(cat_train_imputed)
cat_columns = encoder.get_feature_names_out(cat_train_imputed.columns)
cat_train_imputed_encoded = pd.DataFrame(cat_data, columns=cat_columns)
cat_data = encoder.transform(cat_val_imputed)
cat_columns = encoder.get_feature_names_out(cat_val_imputed.columns)
cat_val_imputed_encoded = pd.DataFrame(cat_data, columns=cat_columns)

X_train = pd.concat([num_train_imputed, cat_train_imputed_encoded], axis=1)

random_forest = RandomForestClassifier(
    n_estimators=100, max_depth=10, random_state=1)
regr = MLPRegressor(random_state=1, max_iter=200000, tol=0.1)
# model = DecisionTreeRegressor(random_state=1)
regr.fit(X_train, y_train)
random_forest.fit(X_train, y_train)
X_val = pd.concat([num_val_imputed, cat_val_imputed_encoded], axis=1)
y_pred_forest = random_forest.predict(X_val)
y_pred_regr = regr.predict(X_val)

print(regr.score(X_val, y_pred_regr))
print(f"MSE:", mean_squared_error(y_val, y_pred_regr))
print(f"MAE:", mean_absolute_error(y_val, y_pred_regr))

print(f"R^2 Score: {r2_score(y_val, y_pred_forest)}")
print(
    f"The MAPE of test dataset is {mean_absolute_percentage_error(y_val, y_pred_forest)}")
print(
    f"The MAE of test dataset is {mean_absolute_error(y_val, y_pred_forest)}")

test_data.head()
test_data_droped = test_data.drop("Id", axis=1, inplace=False)
test_data_droped = test_data[features]
test_data_num = test_data_droped.select_dtypes(exclude="object")
test_data_cat = test_data_droped.select_dtypes(include="object")
imputed_test_data_num = pd.DataFrame(imputer_num.transform(
    test_data_num), columns=test_data_num.columns)
imputed_test_data_cat = pd.DataFrame(imputer_cat.transform(
    test_data_cat), columns=test_data_cat.columns)
encoded_imputed_test_data_cat = pd.DataFrame(encoder.transform(
    imputed_test_data_cat), columns=encoder.get_feature_names_out(imputed_test_data_cat.columns))
X_test = pd.concat(
    [imputed_test_data_num, encoded_imputed_test_data_cat], axis=1)
y_pred = regr.predict(X_test)

result = pd.DataFrame({"Id": test_data["Id"], "SalePrice": y_pred})
print(result.shape)
result.to_csv("./house_price_submissioin.csv", index=False, header=True)
