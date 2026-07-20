import pandas as pd 
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import json
model = LogisticRegression(max_iter=1000)

df = pd.read_csv('./data/Telecom_Tower_Failure_Dataset_10000-1.csv')

X = df.drop(['Tower_ID', 'Failure_Within_48Hrs'], axis=1).values

Y = df['Failure_Within_48Hrs'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=13221)

model.fit(X_train, Y_train)

test_eval = model.score(X_test, Y_test)

train_eval = model.score(X_train, Y_train)


test_predictions = model.predict(X_test)
print(test_eval)

print(train_eval)

roc_auc = roc_auc_score(test_predictions, Y_test)
print()

joblib.dump(model, 'logistic_reg.joblib')
with open('metric_logistic_reg.json', 'w') as f:
    json.dump({"test_accuracy": test_eval, "train_accuracy": train_eval, "roc_auc": roc_auc}, f)

