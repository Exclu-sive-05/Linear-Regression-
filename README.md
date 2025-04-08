# Linear-Regression-
import pandas as pd
import numpy as np
import matplotlip.pyplot as plt
from sklearn import Linear_model
from sklearn.metrics import r2_score
df=pd.read_csv(r"c:\users\chand\Documents\df1.csv")
df
%matplotlib inline
plt.xlabel("x(Size in sqft)")
plt.ylabel("y(Price in lakhs)")
plt.scatter(df.x,df.y,color='red' marker='+')
reg=linear_model.LinearRegression()
reg.fit(df[['x']],df.y)
reg.predict([[700]])
y_pred=reg.predict(df[['x']])
r_squared=r2_score(df[['y']],y_pred)
print(f"R-Squared:{r_squared}")
