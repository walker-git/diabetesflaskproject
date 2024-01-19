import pandas as pd
from flask import Flask,render_template,request
import numpy as np
import pickle

app=Flask(__name__)
model=pickle.load(open('model\\Diabetes_Classifier.pkl','rb'))



@app.route("/")
def homepage():
    return render_template('homepage.html')





@app.route("/predict",methods=['POST'])
def predict():
    pregnancies=float(request.form['PREGNANCIES'])
    glucose=float(request.form['GLUCOSE'])
    bp=float(request.form['BP'])
    skin_thickness=float(request.form['SKIN THICKNESS'])
    insulin=float(request.form['INSULIN'])
    bmi=float(request.form['BMI'])
    DBF=float(request.form['DBF'])
    AGE=float(request.form['AGE'])




    data=[np.array([pregnancies,glucose,bp,skin_thickness,insulin,bmi,DBF,AGE])]
    result=model.predict(data)
    final_prediction=round(result[0],0)


    return render_template('homepage.html',prediction_value="Prediction: "+str(final_prediction))



if __name__=='__main__':
    app.run(debug=False, port=8000) 