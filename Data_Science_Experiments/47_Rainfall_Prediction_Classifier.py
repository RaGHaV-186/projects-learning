import pandas as pd
from sklearn.compose import ColumnTransformer

pd.set_option('display.max_columns',None)
pd.set_option('display.width',1000)
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv"
df = pd.read_csv(url)

print(df.head())

df = df.dropna()

print(df.info())

print(df.columns)

df = df.rename(columns={'RainToday': 'RainYesterday',
                        'RainTomorrow': 'RainToday'
                        })

df = df[df.Location.isin(['Melbourne','MelbourneAirport','Watsonia',])]
df. info()

def date_to_season(date):
    month = date.month
    if (month == 12) or (month == 1) or (month == 2):
        return 'Summer'
    elif (month == 3) or (month == 4) or (month == 5):
        return 'Autumn'
    elif (month == 6) or (month == 7) or (month == 8):
        return 'Winter'
    elif (month == 9) or (month == 10) or (month == 11):
        return 'Spring'


df['Date'] = pd.to_datetime(df['Date'])

df['Season'] = df['Date'].apply(date_to_season)

df = df.drop(columns=['Date'])

print(df.head())
print(df['Season'].value_counts())

X = df.drop(columns=['RainToday'])
y =df['RainToday']

print(f"Features Shape: {X.shape}")
print(f"Target Shape:   {y.shape}")

print("\nFirst row of X:")
print(X.iloc[0])

counts = y.value_counts()

percentages = y.value_counts(normalize=True) * 100

print("--- Class Counts ---")
print(counts)

print("\n--- Class Percentages ---")
print(f"No Rain: {percentages['No']:.2f}%")
print(f"Rain:    {percentages['Yes']:.2f}%")

plt.figure(figsize=(6, 4))
sns.countplot(x=y, hue=y, legend=False, palette='coolwarm')
plt.title("Class Balance: Did it Rain?")
plt.show()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

numeric_features = X_train.select_dtypes(include=['number']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object','category']).columns.tolist()


numeric_transformer = Pipeline(steps=[
    ('scaler',StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot',OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num',numeric_transformer,numeric_features),
        ('cat',categorical_transformer,categorical_features)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

param_grid = {
    'classifier__n_estimators':[50,100],
    'classifier__max_depth':[None,10,20],
    'classifier__min_samples_split':[2,5]
}

cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy',
    verbose=2,
)


grid_search.fit(X_train,y_train)

print("\nBest parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

y_pred = grid_search.predict(X_test)

print("\nClassification Report")
print(classification_report(y_test,y_pred))

conf_matrix = confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap="Blues")
plt.title('Confusion Matrix')
plt.show()

feature_importances = grid_search.best_estimator_['classifier'].feature_importances_

feature_names = numeric_features + list(grid_search.best_estimator_['preprocessor']
                                        .named_transformers_['cat']
                                        .named_steps['onehot']
                                        .get_feature_names_out(categorical_features))

importance_df = pd.DataFrame({'Feature': feature_names,
                              'Importance': feature_importances
                             }).sort_values(by='Importance', ascending=False)

N = 20
top_features = importance_df.head(N)

plt.figure(figsize=(10,6))
plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
plt.gca().invert_yaxis()
plt.title(f'Top {N} Most Important Features in predicting whether it will rain today')
plt.xlabel('Importance Score')
plt.show()

