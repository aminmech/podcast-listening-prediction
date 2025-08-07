
# 🎧 Podcast Listening Time Prediction

This project predicts how long a user will listen to a podcast episode based on episode features like length, number of ads, guest popularity, and more.  
It uses **XGBoost regression** and compares two hyperparameter tuning strategies: **GridSearchCV** and **Optuna**.

---

## 📊 Problem Statement

Podcast platforms want to optimize content layout and personalize suggestions. Predicting how long users engage with content is valuable for:
- Better recommendations
- Ad placement
- Audience retention strategies

---

## 🧠 Modeling Approach

Two methods are applied using XGBoost:
1. `GridSearchCV` — exhaustive manual tuning
2. `Optuna` — Bayesian optimization for efficient tuning

Both use 5-fold cross-validation and are evaluated using RMSE (Root Mean Squared Error).

---

## 📁 File Structure

```bash
📦podcast-listening-prediction
 ┣ 📄 xgboost_gridsearch.py      # XGBoost with GridSearch tuning
 ┣ 📄 xgboost_optuna.py          # XGBoost with Optuna tuning
 ┣ 📄 train.csv                  # Training data (if included)
 ┣ 📄 requirements.txt           # Project dependencies
 ┗ 📄 README.md                  # Project overview
🧪 Dataset Features
Episode_Length_minutes

Genre, Publication_Day, Publication_Time

Host_Popularity_percentage

Guest_Popularity_percentage

Number_of_Ads

Episode_Sentiment

# How to Use
1. Install dependencies:
bash
Copy
Edit
pip install -r requirements.txt
2. Run the models:
For GridSearch:

bash
Copy
Edit
python xgboost_gridsearch.py
For Optuna:

bash
Copy
Edit
python xgboost_optuna.py
Both scripts print:

Best hyperparameters

Validation RMSE

Final prediction output

📈 Sample Output
yaml
Copy
Edit
GridSearchCV:
Best Parameters: {'learning_rate': 0.05, 'max_depth': 7, ...}
Test RMSE: 13.2

Optuna:
Best Parameters: {'learning_rate': 0.07, 'max_depth': 6, ...}
Test RMSE: 12.8
📦 Requirements
text
Copy
Edit
xgboost
scikit-learn
pandas
numpy
optuna
🔮 Future Improvements
Add LightGBM and CatBoost comparisons

Test feature engineering (ratios, bins)

Deploy with FastAPI or Streamlit

👤 Author
Amin Salehi Tabrizi (@aminmech)
📍 Machine Learning Engineer & Computational Scientist
📫 github.com/aminmech

📌 Credits
Kaggle Playground Series S5E4

Libraries: XGBoost, Optuna, scikit-learn





Would you like a matching `requirements.txt` file too?
