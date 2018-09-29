from math import sqrt
from time import process_time
from typing import Dict, TypeVar

from numpy import number
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, auc

from IPython.display import display


# Define custom types to be typehinted for define_model/metric funcs
Model = TypeVar("Model")
Metric = TypeVar("Metric")


def main():
    # # Set index with the 'unnamed' column and assign it a name
    # df.set_index(df.columns[0], inplace=True)
    # df.index.rename("Diamond No.", inplace=True)

    # # Use pandas in-built feature to create integer values for categories
    # df = pd.get_dummies(df, columns=["cut", "color", "clarity"])

    # # Then assign the category dtype to these columns to be labeled for the pipeline processing
    # df[df.columns[1:4]] = df.iloc[:, 1:4].astype("category")

    traindf = pd.read_csv("adult.data")
    testdf = pd.read_csv("adult.test")

    # print(traindf)
    print traindf

    # Normalise our features
    clean_pipeline().fit_transform(df)

    # Finally, train + predict. Record training time and measure prediction against several metrics
    for name, reg in models.items():
        print(f"{name} TRIAL:")

        start_time = process_time()
        trained_model = reg.fit(X_train, y_train)
        print(f"\t{process_time()-start_time :.2f} seconds to train")

        diagnostics(y_pred=trained_model.predict(X_test), y_true=y_test)


""" Runs and prints (five) regression metrics on the model predictions. The metrics are prepared at the start of 
    the program and saved globally.

    Params:
        y_pred (array-like, sparse matrix of floats): regression predictions made by the model
        y_true (pandas Series eof floats): true values 
    
    Returns:
        None
"""
def diagnostics(y_pred, y_true):
    for m_name, metric in metrics.items():
        print(f"\t{m_name}: {metric(y_pred=y_pred, y_true=y_true):.3f}")
    print()


""" Run at start of program to create a dictionary Sequence of regression models """
def define_models() -> Dict[str, Model]:
    model_names = 
        KNeighbors Classifier
        NaiveBayes, Bernouolli
        Support Vector Classifier
        Decision Tree Classifier
        Random Forest Classifier
        AdaBoost Classifier
        Gradient Boosting Classifier
        Linear Discriminant Analysis
        MLP Classifier
        Logistic Regression
        .split()

    regressors = [
        KNeighborsClassifier(),
        BernoulliNB(),
        SVC(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        LinearDiscriminantAnalysis(),
        MLPClassifier(),
        LogisticRegression()
    ]

    return dict(zip(model_names, regressors))


""" Run at start of program to create a dictionary Sequence of regression metrics """
def define_metrics() -> Dict[str, Metric]:
    metric_names = """mean_squared_error
     root_mean_squared_error
     r2_score 
     mean_absolute_error""".split()

    metric_funcs = [
        mean_squared_error,
        root_mean_squared_error,
        r2_score,
        mean_absolute_error,
    ]

    return dict(zip(metric_names, metric_funcs))


# Global code, ran at startup to define what models to run and which metrics to test on the results
models = define_models()
metrics = define_metrics()

""" Use on a pandas DataFrame and apply preprocessing to columns of similar dtypes """
def clean_pipeline():
    return make_pipeline(
        FeatureUnion(
            transformer_list=[
                (
                    "numeric",
                    make_pipeline(
                        TypeSelector(number),
                        SimpleImputer(strategy="median"),
                        StandardScaler(),
                    ),
                ),
                (
                    "category",
                    make_pipeline(
                        TypeSelector("category"),
                        SimpleImputer(strategy="most_frequent"),
                        OneHotEncoder(),
                    ),
                ),
            ]
        )
    )


""" Use in pipeline on a pandas DataFrame to select all columns of a certain dtype """
class TypeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])


if __name__ == "__main__":
    main()