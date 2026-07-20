import pandas as pd 
import joblib


df = pd.read_csv('./data/Telecom_Tower_Failure_Dataset_10000-1.csv').head(10)

model = joblib.load('logistic_reg.joblib')


X = df.drop(['Tower_ID', 'Failure_Within_48Hrs'], axis=1).values
Y = df['Failure_Within_48Hrs'].values

print(model.score(X, Y))

