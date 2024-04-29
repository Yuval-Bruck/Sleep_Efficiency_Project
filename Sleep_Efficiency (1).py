import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)

# Read the SleepEfficiency.cvs file and create the "X" and "y" numpy arrays
data_frame = pd.read_csv('my_data/Sleep_Efficiency.csv')

# Perform one-hot encoding on 'Gender' column
encoder = OneHotEncoder(drop='first', sparse_output=False)
gender_encoded = encoder.fit_transform(data_frame[['Gender']]) # One-hot encode 'Gender'

# Perform one-hot encoding on 'Smoking status' column
smoking_encoder = OneHotEncoder(drop='first', sparse_output=False)
smoking_status_encoded = smoking_encoder.fit_transform(data_frame[['Smoking status']]) # One-hot encode 'Smoking status'

# Display minimum, maximum, and average values for each column
stats = data_frame.describe()
print("Column Statistics:")
print(stats)

# Drop the original 'Gender' column from the DataFrame and concatenate the encoded columns
data_frame = data_frame.drop(columns=['Gender'])
data_frame = pd.concat([data_frame, pd.DataFrame(gender_encoded,
                                                 columns=encoder.get_feature_names_out(['Gender']))], axis=1)

# Drop the original 'Smoking status' column from the DataFrame and concatenate the encoded columns
data_frame = data_frame.drop(columns=['Smoking status'])
data_frame = pd.concat([data_frame,
                        pd.DataFrame(smoking_status_encoded,
                                     columns=smoking_encoder.get_feature_names_out(['Smoking status']))], axis=1)

X = data_frame.iloc[:,:]


# calculate the correlations
correlation = X.corr()

# create the correlations matrix
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix (Sleep efficiency data):')
plt.show()
print("X array")
print(X)

X = data_frame.iloc[:,:].to_numpy()  # Features for model

# Specify numeric columns for correlation computation
numeric_columns = data_frame.select_dtypes(include=np.number).columns
correlation_matrix = data_frame[numeric_columns].corr()

# Print the correlation matrix
print("\nCorrelation Matrix:")
print(correlation_matrix)

first_10_rows = data_frame.head(10)
print("\nfirst 10 rows:\n", first_10_rows)

print("\ndata frame columns:\n", data_frame.columns, "\n")

# Extract target variable and convert to categorical
y = data_frame.iloc[:, -1].to_numpy()
y = to_categorical(y)

print(data_frame)

"""
Gender:
Possible values: Male, Female.
Explanation: Indicates the gender of the individual in the dataset.

Smoking status:
Possible values: Yes, No.
Explanation: Indicates whether the individual smokes or not.
"""

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# features Scaling - Standard scaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print("X_train: \n", x_train)
print("X_test: \n", x_test)

for k in range(1, 10):
    classifier = KNeighborsClassifier(n_neighbors=k)  # The number of neighbors used for classification
    classifier.fit(x_train, y_train.argmax(axis=1))  # argmax to convert one-hot encoded labels to original labels
    y_pred = classifier.predict(x_test)
    accuracy = accuracy_score(y_test.argmax(axis=1), y_pred)
    print(f'k = {k}:')
    print(f"Accuracy = {accuracy}")
    print(f"y_pred = {y_pred}")

    print("y_test =", y_test.argmax(axis=1))

    cm = confusion_matrix(y_test.argmax(axis=1), y_pred)  # Convert one-hot encoded labels to original label
    print("confusion matrix: \n", cm)

# Train the KNN classifier with the best k value and set feature names
classifier_best = KNeighborsClassifier(n_neighbors=1)
classifier_best.fit(x_train, y_train.argmax(axis=1))
classifier_best.feature_names_in_ = list(data_frame.columns)

# Perform prediction with the best k value
y_pred_best = classifier_best.predict(x_test)
cm_best = confusion_matrix(y_test.argmax(axis=1), y_pred_best)

# Print the confusion matrix
print(f'Confusion Matrix (Best k = {1}):')
print(cm_best)


X_feature_names = ['Age', 'Sleep duration', 'Sleep efficiency', 'REM sleep percentage',
                   'Deep sleep percentage', 'Light sleep percentage', 'Awakenings',
                   'Caffeine consumption', 'Alcohol consumption', 'Exercise frequency'
                   'Gender', 'Smoking status']

new_data_point = pd.DataFrame({
    'Age': [65],
    'Sleep duration': [6],
    'Sleep efficiency': [0.88],
    'REM sleep percentage': [18],
    'Deep sleep percentage': [70],
    'Light sleep percentage': [12],
    'Awakenings': [0],
    'Caffeine consumption': [0],
    'Alcohol consumption': [2],
    'Exercise frequency': [3],
    'Gender': ['Female'],
    'Smoking status': ['No']
})

# One-hot encode 'Gender' and 'Smoking status' columns in the new data point
gender_encoded_new = encoder.transform(new_data_point[['Gender']])
smoking_status_encoded_new = smoking_encoder.transform(new_data_point[['Smoking status']])

# Dropping the original 'Gender' and 'Smoking status' columns from the new data point
new_data_point = new_data_point.drop(columns=['Gender', 'Smoking status'])

# Concatenate the encoded columns with the new data point
new_data_point = pd.concat([new_data_point, pd.DataFrame(gender_encoded_new, columns=encoder.get_feature_names_out(['Gender']))], axis=1)
new_data_point = pd.concat([new_data_point, pd.DataFrame(smoking_status_encoded_new, columns=smoking_encoder.get_feature_names_out(['Smoking status']))], axis=1)

# Predict using the trained KNN classifier with the best k value for the new data point
predicted_label_new = classifier_best.predict(new_data_point)

print(f'Predicted Label for New Data Point: {predicted_label_new}')
print(f'Features for Prediction: {new_data_point}')


input_shape = x_train.shape[1]


# 1 - Define the model
model = Sequential([
    Input(shape=(input_shape,)),  # Input layer with the correct input shape
    Dense(16, activation='sigmoid'),
    Dense(8, activation='elu'),
    Dense(4, activation='tanh'),
    Dense(2, activation='softmax')  # Number of labels (binary classification)
])


# 2 - Configure the learning process
model.compile(
    optimizer='Adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 3 - Train the model
history = model.fit(
    x=x_train,
    y=y_train,
    epochs=100,
    shuffle=True
)

# 4 - Evaluate the model
score = model.evaluate(x_train, y_train, verbose=0)
print('Train loss:', score[0])
print('Train accuracy:', score[1])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


model.save('Sleep_Efficiency_Model.keras')  # Saving the model
