from numpy import ravel
from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as knn

import sys
from datetime import datetime

knLow = 10
knHigh = 15

foldsLow = 5
foldsHigh = 10

# fetch dataset
predict_students_dropout_and_academic_success = fetch_ucirepo(id=697)

# data (as pandas dataframes)
orig_X = predict_students_dropout_and_academic_success.data.features
orig_y = predict_students_dropout_and_academic_success.data.targets

# choosing the input variables
columns = [
    'Marital Status',
    'Application order',
    # 'Daytime/evening attendance', # Not Included due to "boolean" value
    'Nacionality',
    'Admission grade',
    # 'Displaced',                  # Not Included due to "boolean" value
    # 'Educational special needs',  # Not Included due to "boolean" value
    # 'Debtor',                     # Not Included due to "boolean" value
    # 'Tuition fees up to date',    # Not Included due to "boolean" value
    # 'Gender',                     # Not Included due to "boolean" value
    # 'Scholarship holder',         # Not Included due to "boolean" value
    'Age at enrollment',
    'International',
    'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)'
]

"""
    3 Choose 8 from columns : 56 permutations

    for each permutation
        for each fold from foldsLow (5) to foldsHigh (10) inclusive (6 different fold values)
            for each kn from knLow (10) to knHigh (15) inclusive    (6 different k-neighbor values)
                run the calculation
                record the highest accuracy of checked k-nearest neighbors
            record the highest accuracy of checked fold counts
        output record
    
    Total Permutations of Columns, Folds, and KNs : 2,016
"""

# Used to store result of each permutation
dtString = datetime.now().strftime("%Y%m%d-%H%M%S")
filename = f"{dtString}.txt"

for index in range(0, len(columns) - 2):
    for jindex in range(index + 1, len(columns) - 1):
        for kindex in range(jindex + 1, len(columns)):
            outputFile = open(filename, 'a')

            input_names = [columns[index], columns[jindex], columns[kindex]]
            num_inputs = len(input_names)
            print("------------------------------", file=outputFile)
            print("Columns: %10s, %10s, %10s" % (input_names[0][:10], input_names[1][:10], input_names[2][:10]), file=outputFile)
            print("Columns: %10s, %10s, %10s" % (input_names[0][:10], input_names[1][:10], input_names[2][:10]))
            
            bestFolds = None
            foldsBestKN = None
            foldsHighAcc = 0
            foldsHighGradAcc = 0
            foldsHighEnrollAcc = 0
            foldsHighDropAcc = 0
            
            for numFolds in range(foldsLow, foldsHigh + 1, 1):
                bestKN = None
                knHighAcc = 0
                knHighGradAcc = 0
                knHighEnrollAcc = 0
                knHighDropAcc = 0

                for kn in range(knLow, knHigh + 1, 1):
                    X = orig_X[input_names]
                    y = orig_y

                    size = X.shape[0]

                    # separating the data into numFolds folds
                    folds = []
                    folds.append(X.iloc[:size//numFolds].values)
                    for temp in range(1, numFolds-1):
                        folds.append(X.iloc[temp*size//numFolds:(temp+1)*size//numFolds].values)
                    
                    # folds = [
                    #     X.iloc[:size//(numFolds+1)].values,
                    #     X.iloc[size//(numFolds+1):2*size//(numFolds+1)].values,
                    #     X.iloc[2*size//(numFolds+1):3*size//(numFolds+1)].values,
                    #     X.iloc[3*size//(numFolds+1):4*size//(numFolds+1)].values
                    # ]

                    folds_y = []
                    folds_y.append(ravel(y.iloc[:size//numFolds].values))
                    for temp in range(1, numFolds-1):
                        folds_y.append(ravel(y.iloc[temp*size//numFolds:(temp+1)*size//numFolds].values))
                    
                    # folds_y = [
                    #     ravel(y.iloc[:size//(numFolds+1)].values),
                    #     ravel(y.iloc[size//(numFolds+1):2*size//(numFolds+1)].values),
                    #     ravel(y.iloc[2*size//(numFolds+1):3*size//(numFolds+1)].values),
                    #     ravel(y.iloc[3*size//(numFolds+1):4*size//(numFolds+1)].values)
                    # ]

                    test_X = X.iloc[(numFolds-1)*size//numFolds:].values
                    test_y = ravel(y.iloc[(numFolds-1)*size//numFolds:].values)

                    neigh = knn(n_neighbors=kn) #, weights='distance')

                    # Initializing some variables used for testing statistics
                    lr_stats = {'Graduate': 0, 'Enrolled': 0, 'Dropout': 0}
                    knn_stats = {'Graduate': 0, 'Enrolled': 0, 'Dropout': 0}
                    total_stats = {'Graduate': 0, 'Enrolled': 0, 'Dropout': 0, 'All': 0}
                    num_correct = 0

                    for validation_num in range(0, numFolds-1):
                        # find the total size of all the training data to determine how big our arrays should be
                        sample_size = 0
                        for a in range(0, numFolds-1):
                            if a != validation_num:
                                sample_size += len(folds[a])

                        X_cur = np.zeros(shape=(sample_size, folds[0].shape[1]))
                        X_validation = folds[validation_num]
                        
                        y_cur = np.zeros(shape=(sample_size,), dtype=object)
                        y_validation = folds_y[validation_num]
                        
                        cur = 0
                        for a in range(0, numFolds-1):
                            if a != validation_num:
                                for b in range(0, len(folds[a])):
                                    # put all the values besides those in the ith fold into the X_cur array
                                    X_cur[cur] = folds[a][b]
                                    y_cur[cur] = folds_y[a][b]
                                    cur += 1

                        # The actual knn function goes here
                        neigh.fit(X_cur, y_cur)

                        total_stats['All'] += len(folds[validation_num])
                        for a in range(0, len(X_validation)):
                            total_stats[y_validation[a]] += 1

                            if neigh.predict(X_validation[a].reshape(1,-1)) == y_validation[a]:
                                num_correct += 1
                                knn_stats[y_validation[a]] += 1

                    iterAcc = float(num_correct)/total_stats['All']
                    iterDropAcc = float(knn_stats['Dropout'])/total_stats['Dropout']
                    iterEnrollAcc = float(knn_stats['Enrolled'])/total_stats['Enrolled']
                    iterGradAcc = float(knn_stats['Graduate'])/total_stats['Graduate']
                    
                    if iterAcc > knHighAcc:
                        knHighAcc = iterAcc
                        knHighGradAcc = iterGradAcc
                        knHighEnrollAcc = iterEnrollAcc
                        knHighDropAcc = iterDropAcc
                        bestKN = kn

                if knHighAcc > foldsHighAcc:
                    foldsHighAcc = knHighAcc
                    foldsHighGradAcc = knHighGradAcc
                    foldsHighEnrollAcc = knHighEnrollAcc
                    foldsHighDropAcc = knHighDropAcc
                    bestFolds = numFolds
                    foldsBestKN = bestKN

            print(f"Folds: {bestFolds}", file=outputFile)
            print(f"KNN: {foldsBestKN}", file=outputFile)
            print(f"Accuracy: {foldsHighAcc*100:2.2f}%", file=outputFile)
            print(f"Dropout Accuracy: {foldsHighGradAcc*100:2.2f}%", file=outputFile)
            print(f"Enrolled Accuracy: {foldsHighEnrollAcc*100:2.2f}%", file=outputFile)
            print(f"Graduate Accuracy: {foldsHighDropAcc*100:2.2f}%\n", file=outputFile)

            outputFile.close() # Closing it to clear the buffer
