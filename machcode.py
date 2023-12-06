#importing required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn import ensemble
from scipy import stats
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

df=pd.read_csv(r"tj\waze_dataset.csv")
df.isnull().sum()
df.dropna(subset=['label'], inplace=True)



columns_with_zeros = (df == 0).any()
num_columns_with_zeros = columns_with_zeros.sum()
columns_with_zeros_names = columns_with_zeros[columns_with_zeros].index.tolist()


column_mean1 = df['sessions'].mean()
column_mean2 = df['drives'].mean()
column_mean3 = df['total_navigations_fav1'].mean()
column_mean4 = df['total_navigations_fav2'].mean()
column_mean5 = df['activity_days'].mean()
column_mean6 = df['driving_days'].mean()
df["sessions"] = df["sessions"].replace(0, column_mean1)
df["drives"] = df["drives"].replace(0, column_mean2)
df["total_navigations_fav1"] = df["total_navigations_fav1"].replace(0, column_mean3)
df["total_navigations_fav2"] = df["total_navigations_fav2"].replace(0, column_mean4)
df["activity_days"] = df["activity_days"].replace(0, column_mean5)
df["driving_days"] = df["driving_days"].replace(0, column_mean6)

class_counts = df['label'].value_counts()
imbalance_ratio = class_counts['retained'] / class_counts['churned']
# Separate the majority and minority classes
majority_class = df[df['label'] == 'retained']
minority_class = df[df['label'] == 'churned']
# Determine the number of additional samples needed to balance the classes
num_samples_needed = len(majority_class) - len(minority_class)
# Randomly sample from the minority class to create synthetic data
oversampled_minority = minority_class.sample(n=num_samples_needed, replace=True, random_state=42)
# Combine the majority class with the oversampled minority class
balanced_df = pd.concat([majority_class, oversampled_minority], axis=0)
# Shuffle the balanced dataset
balanced_df = balanced_df.sample(frac=1, random_state=42)
class_counts = balanced_df['label'].value_counts()


Xdata = balanced_df.drop(['label'], axis=1)
ydata = balanced_df['label']

yenc = np.asarray([1 if c == 'Retained' else 0 for c in ydata])
cols = ['drives', 'total_sessions','total_navigations_fav1','total_navigations_fav2','driven_km_drives','duration_minutes_drives']
Xdata = balanced_df[cols]


X_train, X_test, y_train, y_test = train_test_split(Xdata, yenc, 
                                                    test_size=0.2,
                                                    random_state=43)
print('Shape training set: X:{}, y:{}'.format(X_train.shape, y_train.shape))
print('Shape test set: X:{}, y:{}'.format(X_test.shape, y_test.shape))

model = ensemble.RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Accuracy : {}'.format(accuracy_score(y_test, y_pred)))
joblib.dump(model, 'model.pkl')
# Making predictions on the test data
y_pred = model.predict(X_test)

# Evaluating the model and calculating metrics
accuracy_random = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(accuracy_random)
print(confusion_mat)
print(classification_rep)



