import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_good = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8],
    'Exam_Score':    [15, 25, 35, 45, 55, 65, 75, 85]
}
df_good = pd.DataFrame(data_good)

data_bad = {
    'Car_Speed':      [10, 20, 30, 40, 50, 60, 70, 80],
    'Fuel_Efficiency': [10, 20, 30, 35, 30, 20, 10, 5]
}
df_bad = pd.DataFrame(data_bad)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.regplot(x="Hours_Studied",y="Exam_Score",data=df_good,color="blue")
plt.title('Good Fit: Regression Plot')

plt.subplot(1,2,2)
sns.residplot(x="Hours_Studied",y="Exam_Score",data=df_good,color="green")
plt.title('Good Fit: Residual Plot')

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.regplot(x='Car_Speed', y='Fuel_Efficiency', data=df_bad, color='red')
plt.title('Bad Fit: Regression Plot')

plt.subplot(1, 2, 2)
sns.residplot(x='Car_Speed', y='Fuel_Efficiency', data=df_bad, color='orange')
plt.title('Bad Fit: Residual Plot (Pattern!)')

plt.tight_layout()
plt.show()

