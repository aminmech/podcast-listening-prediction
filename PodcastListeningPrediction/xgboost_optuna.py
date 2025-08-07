import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import optuna


# Load and preprocess training data
df = pd.read_csv(r"C:\Users\KLP\Desktop\MLcourse\train.csv",index_col=0)
df = df.drop(["Podcast_Name", "Episode_Title"], axis=1)

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype("category")

X = df.drop("Listening_Time_minutes", axis=1)
Y = df["Listening_Time_minutes"]

# Define Optuna objective with 5-fold CV
def objective(trial):
    params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "enable_categorical": True,
        "max_depth": trial.suggest_categorical("max_depth", [3, 5, 7]),
        "learning_rate": trial.suggest_categorical("learning_rate", [0.05, 0.1]),
        "n_estimators": trial.suggest_categorical("n_estimators", [100, 200]),
        "subsample": trial.suggest_categorical("subsample", [0.8, 1.0]),
        "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.8, 1.0])
    }

    model = xgb.XGBRegressor(**params)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, Y, cv=kf, scoring='neg_root_mean_squared_error')
    return -scores.mean()

# Match GridSearch total evaluations: 3*2*2*2*2 = 48
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=48)

# Train best model on full data
best_params = study.best_params
best_params.update({
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "enable_categorical": True
})

best_model = xgb.XGBRegressor(**best_params)
best_model.fit(X, Y)

# Load and preprocess test data
df_test2 = pd.read_csv(r"C:\Users\KLP\Desktop\MLcourse\test.csv", index_col=0)
for col in df_test2.select_dtypes(include="object").columns:
    df_test2[col] = df_test2[col].astype("category")

X_test2 = df_test2.drop(["Podcast_Name", "Episode_Title"], axis=1)

# Predict and save submission
y_pred2 = best_model.predict(X_test2)
submission_predict = pd.DataFrame({"id": X_test2.index, "Listening_Time_minutes": y_pred2})
submission_predict.to_csv("submission_predict.csv", index=False)

