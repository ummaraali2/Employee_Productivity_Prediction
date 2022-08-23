
#importing libraries
import os
import numpy as np
import pandas as pd
import flask
import pickle
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#from sklearn.ensemble import RandomForestClassifier
##creating instance of the class

app = Flask(__name__)
loaded_model = pickle.load(open("model.pkl","rb"))

#to tell flask what url shoud trigger the function index()
@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')
   

#prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,13)
    #loaded_model = pickle.load(open("model.pkl","rb"))
    result = loaded_model.predict(to_predict)
    return result[0]

@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)
        return render_template("result.html",prediction=result)


if __name__ == "__main__":
    
	app.run(debug=True)