import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report
import itertools
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("creditcard.csv")

# count_classes = pd.value_counts(data['Class'], sort=True).sort_index()
# count_classes.plot(kind='bar')
# plt.title("Fraud class histogram")
# plt.xlabel("Class")
# plt.ylabel("Frequency")
# plt.show()

# fraud_data = data[data.Class == 1]
# print(sum(fraud_data['Amount'].values.ravel()) / len(fraud_data))
# normal_data = data[data.Class == 0]
# print(sum(normal_data['Amount'].values.ravel()) / len(normal_data))
# print(len(normal_data))
# exit()

data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
data = data.drop(['Time','Amount'],axis=1)
# print(data.head())


X = data.iloc[:, data.columns != 'Class']
Y = data.iloc[:, data.columns == 'Class']

# Number of data points in the minority class
number_records_fraud = len(data[data.Class == 1])
fraud_indices = np.array(data[data.Class == 1].index)

total = len(data)
print('total: ' + str(total))
print('fraud: ' + str(number_records_fraud))
step = round((total-number_records_fraud)/10)*1
print(step)


# Picking the indices of the normal classes
normal_indices = data[data.Class == 0].index

# Out of the indices we picked, randomly select "x" number (number_records_fraud)
random_normal_indices = np.random.choice(normal_indices, step, replace=False)
random_normal_indices = np.array(random_normal_indices)


# Appending the 2 indices
under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])
# np.random.shuffle(under_sample_indices)
under_sample_indices = np.sort(under_sample_indices)

# Under sample dataset
under_sample_data = data.iloc[under_sample_indices,:]
# print(under_sample_data)

X_undersample = under_sample_data.iloc[:, under_sample_data.columns != 'Class']
Y_undersample = under_sample_data.iloc[:, under_sample_data.columns == 'Class']

# Showing ratio
print("Percentage of normal transactions: ", len(under_sample_data[under_sample_data.Class == 0])/len(under_sample_data))
print("Percentage of fraud transactions: ", len(under_sample_data[under_sample_data.Class == 1])/len(under_sample_data))
print("Total number of transactions in resampled data: ", len(under_sample_data))

# Whole dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# Undersampled dataset
X_train_undersample, X_test_undersample, Y_train_undersample, Y_test_undersample = train_test_split(X_undersample
                                                                                                   ,Y_undersample
                                                                                                   ,test_size=0.3
                                                                                                   ,random_state=0)


def printing_Kfold_scores(x_train_data, y_train_data):
    fold = KFold(5)

    # Different C parameters
    c_param_range = [0.01, 0.1, 1, 10, 100]

    results_table = pd.DataFrame(index=range(len(c_param_range), 2), columns=['C_parameter', 'Mean recall score'])
    results_table['C_parameter'] = c_param_range

    # the k-fold will give 2 lists: train_indices = indices[0], test_indices = indices[1]
    j = 0
    for c_param in c_param_range:
        print('-------------------------------------------')
        print('C parameter: ', c_param)
        print('-------------------------------------------')
        print('')

        recall_accs = []
        for iteration, indices in enumerate(fold.split(y_train_data), start=1):
            # Call the logistic regression model with a certain C parameter
            lr = LogisticRegression(C=c_param, penalty='l1')

            # Use the training data to fit the model. In this case, we use the portion of the fold to train the model
            # with indices[0]. We then predict on the portion assigned as the 'test cross validation' with indices[1]
            lr.fit(x_train_data.iloc[indices[0], :], y_train_data.iloc[indices[0], :].values.ravel())

            # Predict values using the test indices in the training data
            y_pred_undersample = lr.predict(x_train_data.iloc[indices[1], :].values)

            # Calculate the recall score and append it to a list for recall scores representing the current c_parameter
            recall_acc = recall_score(y_train_data.iloc[indices[1], :].values, y_pred_undersample)
            recall_accs.append(recall_acc)
            print('Iteration ', iteration, ': recall score = ', recall_acc)

        # The mean value of those recall scores is the metric we want to save and get hold of.
        results_table.ix[j, 'Mean recall score'] = np.mean(recall_accs)
        j += 1
        print('')
        print('Mean recall score ', np.mean(recall_accs))
        print('')

    best_c = results_table.loc[results_table['Mean recall score'].idxmax()]['C_parameter']

    # Finally, we can check which C parameter is the best amongst the chosen.
    print('*********************************************************************************')
    print('Best model to choose from cross validation is with C parameter = ', best_c)
    print('*********************************************************************************')

    return best_c

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Матрица неточностей',
                          cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        1#print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# best_c = printing_Kfold_scores(X_train,Y_train)
# lr = LogisticRegression(C = best_c, penalty = 'l1')
# lr.fit(X_train,Y_train.values.ravel())
# y_pred = lr.predict(X_test.values)

print(len(X_train))
print(len(X_test) - Y_test)

dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train.values.ravel())
y_pred = dt.predict(X_test.values)

# lr = LogisticRegression(C=1.0, penalty='l1')
# lr.fit(X_train_undersample,Y_train_undersample.values.ravel())
# y_pred = lr.predict(X_test.values)

# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_test,y_pred)
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))
print("Precision metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[0,1]+cnf_matrix[1,1]))

# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()
