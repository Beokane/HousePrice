from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import GridSearchCV
train_data = pd.read_csv(
    "./house-prices-advanced-regression-techniques/train.csv")
test_data = pd.read_csv(
    "./house-prices-advanced-regression-techniques/test.csv")

# features = ["LotArea", "Street", "YearBuilt", "LandSlope", "HouseStyle", "BedroomAbvGr", "KitchenQual", "GarageArea", "GarageQual"]
# train_data = train_data[features]
# features = ["LotArea", "BedroomAbvGr", "GarageArea"]
# features = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
# train_data.dropna(inplace=True)
drop_features = ['SalePrice', 'Id']
X = train_data.drop(drop_features, axis=1, inplace=False)
y = train_data["SalePrice"]

X_dropped = X.dropna(thresh=0.7*len(X), axis=1, inplace=False)
# print(f"{X.columns[X.isnull().mean() >= 0.2].tolist()}")

X_train, X_val, y_train, y_val = train_test_split(
    X_dropped, y, test_size=0.2, random_state=42)
num_cols = X_train.select_dtypes(exclude="object").columns
cat_cols = X_train.select_dtypes(include="object").columns

num_transformer = Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)],
    remainder='passthrough',
    n_jobs=-1
)

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', MLPRegressor(hidden_layer_sizes=100,
     random_state=1, max_iter=200000, tol=0.001))],
    verbose=True
)

param_grid = {
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    'preprocessor__cat__imputer__strategy': ['most_frequent','constant'],
    'model__hidden_layer_sizes': [(100), (100, 100)],
    'model__activation': ['relu', 'tanh'],
    'model__solver': ['adam', 'sgd']
}

grid_search = GridSearchCV(full_pipeline, param_grid, cv=2, scoring='neg_mean_absolute_error', verbose=2, n_jobs=-1)

grid_search.fit(X_train, y_train)
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: ", -grid_search.best_score_)
best_model = grid_search.best_estimator_
y_val_pred = best_model.predict(X_val)

# full_pipeline.fit(X_train, y_train)
# y_val_pred = full_pipeline.predict(X_val)
# print(full_pipeline.score(X_val, y_val_pred))

print(f"MSE:", mean_squared_error(y_val, y_val_pred))
print(f"R^2 Score: {r2_score(y_val, y_val_pred)}")
print(
    f"The MAPE of test dataset is {mean_absolute_percentage_error(y_val, y_val_pred)}")
print(f"The MAE of test dataset is {mean_absolute_error(y_val, y_val_pred)}")

X_test = test_data.drop("Id", axis=1, inplace=False)
# y_test_pred = full_pipeline.predict(X_test)
y_test_pred = best_model.predict(X_test)
result = pd.DataFrame({"Id": test_data["Id"], "SalePrice": y_test_pred})
print(result.shape)
result.to_csv("./house_price_submissioin.csv", index=False, header=True)
