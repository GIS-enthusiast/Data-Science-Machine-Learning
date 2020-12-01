## conda cheatsheet
https://docs.conda.io/projects/conda/en/latest/_downloads/843d9e0198f2a193a3484886fa28163c/conda-cheatsheet.pdf

## setting up the conda environment
1. First time (in terminal): conda create --prefix ./env pandas numpy matplotlib scikit-learn
2. Or import yml file: conda create --prefix ./env environment.yml (having exported a yml file to the same directory, eg. conda export env > environment.yml)
3.

## Tip to download a library using conda in a Jupyter Notebook.
```py
import sys
#!conda install --yes --prefix {sys.prefix} <module>
```
## Good example for tuning hyperparameters
https://colab.research.google.com/drive/1ISey96a5Ag6z2CvVZKVqTKNWRwZbZl0m#scrollTo=sSoaPfQFoh1t

# sklearn - tabular data machine learning
## Common classes and functions
```py
import sklearn
sklearn.show_versions()

from sklearn.datasets import load_boston

from sklearn.emsemble import RandomForestClassifier
from sklearn.emsemble import RandomForestRegressor
from sklearn.svm import LinearSVC

from sklearn.model_selection import train_test_split

# Metrics documentation: https://scikit-learn.org/stable/modules/model_evaluation.html
# classification metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import absolute_mean_error
from sklearn.model_selection import cross_val_score
## binary metrics
from sklearn.metrics import roc_curve # fpr, tpr, thresholds = roc_curve(y_test, y_probs_positive)
from sklearn.metrics import roc_auc_score # gives area under the roc_curve.
from sklearn.metrics import confusion_matrix
# note that from sklearn.metrics import plot_confusion_matrix is not working at the moment. Two engineered solutions are provided below in graphs section.
from sklearn.metrics import classification_report

# regression metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# preprocessing
from sklearn.preprocessing import OneHotEncoder
pd.get_dummies()
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
imputer.fit_transform()
imputer.transform()

np.random.seed(42)

# create data
df.pop()
df.drop('column', axis=1)
df.isna().sum()
df.dropna(inplace=True)

# model attributes
model.get_params()
model.fit()
model.predict()
model.predict_proba()
model.score()

df.shape

```
## Tuning Hyperparameters: RandomizedSearchCV and GridSearchCV
```py

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV, train_test_split

example_grid = {"n_estimators": [10, 100, 200, 500, 1000, 1200], 
        "max_depth": [None, 5, 10, 20, 30], 
        "max_features":['auto', 'sqrt'], 
       "min_samples_split": [2, 4, 6],
       "min_samples_leaf": [1, 2, 4]}

# identify best parameters
rs_model.best_params_


       
def evaluate_preds(y_true, y_preds):
    '''
    Perfoms evaluation comparison on y_true labels vs. y_preds labels on a classification.
    '''
    accuracy = accuracy_score(y_true, y_preds)
    precision = precision_score(y_true, y_preds)
    recall = recall_score(y_true, y_preds)
    f1 = f1_score(y_true, y_preds)
    metric_dict = {"accuracy": round(accuracy, 2), 
                   "precision": round(precision, 2),
                  "recall": round(recall, 2),
                  "f1": round(f1, 2)
                  }
    print(f'Acc: {accuracy * 100:.2f}%')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1: {f1:.2f}')
    
    return metric_dict


# improve a model 
## try different hyperparameters
np.random.seed(42)
for i in range(10, 100, 10):
    print(f"Trying model with {i} estimators...")
    clf = RandomForestClassifier(i).fit(x_train, y_train)
    print(f"Model accuracy on test set: {clf.score(x_test, y_test) *100:.2f}%")
    print("")
```
## Putting it all together in a sklearn Pipeline:
More resource here: https://colab.research.google.com/drive/1AX3Llawt0zdjtOxaYuTZX69dhxwinFDi?usp=sharing
```py
# Getting data ready
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer # filling missing data
from sklearn.preprocessing import OneHotEncoder # strings to numerical

# Modelling
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

# Set up Random Seed
import numpy as np

# Import data and drop rows with missing labels
data = pd.read_csv("car-sales-extended-missing-data.csv")
data.dropna(subset=['Price'], inplace=True)

# Define different features and transformer pipeline
categorial_features = ['Make', 'Colour']
categorical_transformer = Pipeline(steps=[
                                        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                        ('onehot', OneHotEncoder(handle_unknown='ignore'))
                                         ])

door_feature = ['Doors']
door_transformer = Pipeline(steps=[
                                ('imputer', SimpleImputer(strategy='constant', fill_value=4))
                                  ])

numeric_features = ['Odometer (KM)']
numeric_transformer = Pipeline(steps=[
                                    ('imputer', SimpleImputer(strategy='mean'))
                                     ])

# Setup preprocessing steps
preprocessor = ColumnTransformer(transformers=[('cat', categorical_transformer, categorial_features), 
                                               ('door', door_transformer, door_feature),
                                               ('num', numeric_transformer, numeric_features)
                                              ])

# Creating a preprocessing and modelling pipeline
model = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('model', RandomForestRegressor())
                       ])

# Split data
X = data.drop('Price', axis=1)
y = data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit and score the model
model.fit(X_train, y_train)
model.score(X_test, y_test)

## NEXT CELL

# Use gridsearch CV with pipeline to improve the model
pipe_grid = {
    "preprocessor__num__imputer__strategy": ["mean", "median"],
    "model__n_estimators" : [100, 1000],
    "model__max_depth": [None, 5],
    "model__max_features": ["auto"],
    "model__min_samples_split": [2, 4]
}

gs_model = GridSearchCV(model, pipe_grid, cv=5, verbose=2)
gs_model.fit(X_train, y_train)
gs_model.score(X_test, y_test)
```
## Exporting and Importing a Model/Classifier/Algorithm
1. Pickle
```py
#save a module and load it
import pickle

pickle.dump(clf, open("random_forest_model_1.pk1", "wb"))

loaded_model = pickle.load(open("random_forest_model_1.pk1", "rb"))
loaded_model.score(x_test, y_test) 
```
2. Joblib
```py
from joblib import dump, load

# Save model to file
dump(gs_model, filename="random_forest_model_1.joblib")

# Import a saved joblib model
loaded_job_model = load(filename="random_forest_model_1.joblib")
```
## Helpful Pandas functions
```py
#shuffle the data
heart_disease_shuffled = heart_disease.sample(frac=1)

# compare colums with crosstab
pd.crosstab(df.target, df.sex) # column names 'target' and 'sex'.

# Make a correlation matrix
df.corr()
# Make a correlation matrix more readable
fig, ax = plt.subplots(figsize=(15, 10))
ax = sns.heatmap(corr_matrix, 
                 annot=True, 
                 linewidths=0.5, 
                 fmt=".2f",
                cmap="icefire")
```
## Helpful Numpy functions
```py
np.logspace(-4, 4, 20)
np.arange(10, 1000, 50) #faster than python range()
```
## Other helpful EDA functions
```py
# super quick and helpful visualisation for missing data.
import missingno as msno
msno.bar(df)
```
## matplotlib.pyplot Graph functions for EDA and Metrics
```py
#1. ROC curve
# import matplotlib.pyplot as plt
def plot_roc_curve(fpr, tpr):
    '''
    Plots a roc curve given the false positive rate (fpr) and the true positive rate (tpr) of model.
    '''
    # plot the roc curve
    plt.plot(fpr, tpr, color="orange", label="ROC")
    # plot line with no predictive power (baseline)
    plt.plot([0, 1], [0, 1], color="darkblue", linestyle="--", label="Guessing")
    #custimise the plot 
    plt.xlabel("False Positive Rate (fpr)")
    plt.ylabel("True Positive Rate (tpr)")
    plt.title("Receiver Operating Characteristics (ROC) Curve")
    plt.legend()
    plt.show()

#2a. Confusion Matrix
# visualise confustion matrix with pd.crosstab()
pd.crosstab(y_test,
           y_preds,
           rownames=['Actual Labels'],
           colnames=['Predicted Labels'])
           
#2b. Confusion Matric
# Or make the confustion matrix more visual with seaborn heatmap
import seaborn as sns

# Set the font scale
sns.set(font_scale=1.5)

# Create a confusion matrix
conf_mat = confusion_matrix(y_test, y_preds)

def plot_conf_mat(conf_mat):
    '''
    Plots a confustion matrix using seaborns heatmap()
    '''
    fig, ax = plt.subplots(figsize=(3,3))
    ax = sns.heatmap(conf_mat, 
                    annot=True,
                    cbar=False) # annotate the boxes with conf_mat info
    
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

plot_conf_mat(conf_mat)

# Scatter Plot, EDA
plt.figure(figsize=(10, 6))

# scatter with positive examples (ie with heart disease)
plt.scatter(df.age[df.target==1],
           df.thalach[df.target==1],
           color="salmon")

# scatter with negative examples (no heart disease)
plt.scatter(df.age[df.target==0],
           df.thalach[df.target==0],
           color="lightblue")

plt.title("Heart Disease as a function of Age and Max Heart Rate")
plt.xlabel("Age")
plt.ylabel("Max Heart Rate")
plt.legend(["Disease", "No Disease"])
```
## Function to fit and score models
```py
# put models in dict to create function

models = {"Logistic Regression": LogisticRegression(),
         "KNN": KNeighborsClassifier(),
         "Random Forest": RandomForestClassifier()}

# create function to fit and score models

def fit_and_score(models, X_train, y_train, X_test, y_test):
    '''
    Fits and evaluates given machine learning models.
    models: a dict of machine learning models
    X_train: training data no labels
    X_test: testing data no labels
    y_train: training labels
    y_test: test labels
    '''
    #set random seed
    np.random.seed(42)
    # make dict to record model scores
    model_scores = {}
    # loop through models
    for name, model in models.items():
        # fit model to data
        model.fit(X_train, y_train)
        # evaluate model and append scores to model_scores
        model_scores[name] = model.score(X_test, y_test)
    return model_scores
```
