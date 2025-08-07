import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import time


df=pd.read_csv(r"C:\Users\KLP\Desktop\MLcourse\train.csv",index_col=0)
df=df.drop(["Podcast_Name","Episode_Title"],axis=1)
print(df["Listening_Time_minutes"].mean())

for col in df.select_dtypes(include="object").columns:
     df[col] = df[col].astype("category")
#print(df.isnull().sum())
#print(df.info())

print(df.dtypes)
#df.plot(x="id" , y="Listening_Time_minutes" , kind="line")
#plt.show()

X=df.drop("Listening_Time_minutes",axis=1)
Y=df["Listening_Time_minutes"]

X_train, X_test,Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=42)

#print(X_test.head())

model = xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist" , enable_categorical=True , n_estimators=100 , max_depth=4 , learning_rate=0.1 )
model.fit(X_train,Y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(Y_test,y_pred)
rmse = mse ** 0.5
print(f"RMSE: {rmse:.2f}")

#Define hyperparameter grid
param_grid = {
     'max_depth': [3, 5, 7],
     'learning_rate': [0.05, 0.1],
     'n_estimators': [100, 200],
     'subsample': [0.8, 1.0],
     'colsample_bytree': [0.8, 1.0]
 }

#Run GridSearchCV with 5-fold cross-validation
grid = GridSearchCV(
     estimator=xgb.XGBRegressor(
         objective='reg:squarederror',
         enable_categorical=True,
         tree_method='hist',
         random_state=42
     ),
     param_grid=param_grid,
     scoring='neg_root_mean_squared_error',
     cv=5,
     verbose=1
 )

grid.fit(X, Y)
best_model = grid.best_estimator_
best_rmse = -grid.best_score_

print("\nGrid Search Results:")
print("Best Parameters:")
for param, value in grid.best_params_.items():
     print(f"  {param}: {value}")
print(f"Best Cross-Validated RMSE: {best_rmse:.2f}")

df_test2=pd.read_csv(r"C:\Users\KLP\Desktop\MLcourse\test.csv",index_col=0)

for col in df_test2.select_dtypes(include="object").columns:
     df_test2[col] = df_test2[col].astype("category")

X_test2=df_test2.drop(["Podcast_Name","Episode_Title"],axis=1)
X_test2 = df_test2.drop(columns=["Podcast_Name", "Episode_Title"], errors="ignore")
#print(X_test2.head())

y_pred2 = best_model.predict(X_test2)

submission_predict = pd.DataFrame({"id":X_test2.index , "Listening_Time_minutes":y_pred2 })

submission_predict.to_csv("submission_predict.csv", index=False)


