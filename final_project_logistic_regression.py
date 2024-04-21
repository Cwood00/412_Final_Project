from numpy import ravel
from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from datetime import datetime

numFolds = 5
include_enrolled = False

if include_enrolled:
    raw_data = fetch_ucirepo(id=697).data
    orig_X = raw_data.features
    orig_y = raw_data.targets
else:
    raw_data = fetch_ucirepo(id=697).data.original
    raw_data = raw_data.loc[raw_data['Target'] != 'Enrolled']
    orig_y = raw_data['Target']
    del raw_data['Target']
    orig_X = raw_data

# choosing the input variables
input_names = [
    'Marital Status',
    'Daytime/evening attendance',
    'Previous qualification',
    'Nacionality',
    'Mother\'s qualification',
    'Father\'s qualification',
    'Admission grade',
    'Displaced',
    'Educational special needs',
    'Debtor',
    'Tuition fees up to date',
    'Gender',
    'Scholarship holder',
    'Age at enrollment',
    'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)'
]
# Used to store result of each permutation
dtString = datetime.now().strftime("%Y%m%d-%H%M%S")
filename = f"{dtString}.txt"

outputFile = open(filename, 'a')

X = orig_X[input_names]
y = orig_y

size = X.shape[0]

# separating the data into training and testing data folds
training_X = X.iloc[:(numFolds - 1) * size // numFolds].values
training_y = ravel(y.iloc[:(numFolds - 1) * size // numFolds].values)

test_X = X.iloc[(numFolds - 1) * size // numFolds:].values
test_y = ravel(y.iloc[(numFolds - 1) * size // numFolds:].values)

# Initializing some variables used for testing statistics
correct_predictions = {'Graduate': 0, 'Enrolled': 0, 'Dropout': 0, 'All': 0}
total_predictions = {'Graduate': 0, 'Enrolled': 0, 'Dropout': 0, 'All': 0}

start_time = datetime.now()
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(training_X, training_y)
end_time = datetime.now()
print(f"Took {(end_time - start_time).total_seconds()} seconds to train logistic regression, using {len(training_y)} samples", file=outputFile)

start_time = datetime.now()
predictions = model.predict(test_X)
end_time = datetime.now()
print(f"Took {(end_time - start_time).total_seconds()} seconds to test logistic regression, using {len(test_X)} samples", file=outputFile)

for i in range(len(predictions)):
    prediction = predictions[i]
    true_value = test_y[i]

    total_predictions[true_value] += 1
    total_predictions['All'] += 1

    if prediction == true_value:
        correct_predictions[true_value] += 1
        correct_predictions['All'] += 1

print(f"Total testing accuracy = {(correct_predictions['All'] / total_predictions['All']) * 100}%", file=outputFile)
print(f"Graduate testing accuracy = {(correct_predictions['Graduate'] / total_predictions['Graduate']) * 100}%", file=outputFile)
if include_enrolled:
    print(f"Enrolled testing accuracy = {(correct_predictions['Enrolled'] / total_predictions['Enrolled']) * 100}%", file=outputFile)
print(f"Dropout testing accuracy = {(correct_predictions['Dropout'] / total_predictions['Dropout']) * 100}%", file=outputFile)

outputFile.close()  # Closing it to clear the buffer