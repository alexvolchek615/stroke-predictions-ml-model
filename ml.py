from flask import Flask, render_template, request
import time
import random
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import numpy as np
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def fitModel():
    df = pd.read_csv("train_strokes.csv")
    df = df.drop(['id'], axis=1)

    df = df.drop(df[df['gender'] == 'Other'].index)

    df = df.drop('smoking_status', axis = 1)
    df = df.drop('bmi', axis = 1)

    oldX, y = df.drop('stroke', axis=1).values, df['stroke'].values

    encoder = OneHotEncoder(handle_unknown='ignore')
    ct = ColumnTransformer([('encoder', encoder, [0, 4, 5, 6])], remainder='passthrough')
    X = ct.fit_transform(oldX)

    smote_enn = SMOTEENN(random_state=0)
    X_resampled, y_resampled = smote_enn.fit_resample(X, y)

    smote_tomek = SMOTETomek(random_state=0)
    X_resampled, y_resampled = smote_tomek.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled)

    pipe = Pipeline([('lr', LinearSVC(C=0.0001, class_weight={1: 1.5}))]).fit(X_train, y_train)
    return ct, pipe