from flask import Flask, render_template, request
import time
import random
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
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
import ml
from joblib import dump, load


# Init flask application
app = Flask(__name__)


#ct, pipe = ml.fitModel()

# load notebook here
pipe = load('savedModel.joblib') 
ct = load('savedColumnTransformer.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['POST'])
def prediction():
    gender = request.form['gender']
    age = float(request.form['age'])
    hypertension = int(request.form['hypertension'])
    heartdisease = int(request.form['heartdisease'])
    ever_married = request.form['ever_married']
    work_type = request.form['work_type']
    Residence_type = request.form['Residence_type']
    avg_glucose_level = float(request.form['avg_glucose_level'])


    answer = predictAnswer(ct, pipe, gender, age, hypertension, heartdisease, ever_married, work_type, Residence_type, avg_glucose_level)
    return render_template('prediction.html', answer=answer)


def predictAnswer(ct, pipe, gender, age, hypertension, heartdisease, ever_married, work_type, Residence_type, avg_glucose_level): 
    row = [gender, age, hypertension, heartdisease, ever_married, work_type, Residence_type, avg_glucose_level]
    row = ct.transform([row])
    result = pipe.predict(row)
    result = result[0]
    if result == 1:
        return "Yes, you may be likely to have a stroke. Please see a doctor."
    elif result == 0:
        return "No, you are not likely to have a stroke" 
    else:
        return "A problem has occured in processing your data. Please reach out to Alex. She promises it was working before :("
    

if __name__ == '__main__':
    # you can run init / fiting here
    app.run(debug=True)