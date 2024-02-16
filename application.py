from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np


app = Flask(__name__)

model = pickle.load(open("LinearRegressionModelNew.pkl",'rb'))
bikes = pd.read_csv("CleanedDataNew.csv")


@app.route('/')
def index():
    brand = sorted(bikes['brand'].unique())
    bike_name = sorted(bikes['bike_name'].unique())
    age = sorted(bikes['age'].unique())
    OwnerType = sorted(bikes['Owner_Type'].unique())
    brand.insert(0,"Select Brands ")

    return render_template('home.html', brand=brand, bike_name=bike_name, age=age, Owner_Type=OwnerType)


@app.route('/predict', methods = ['POST'])
def predict():
    brand = request.form.get('brand')
    bike_name = request.form.get('bike_name')
    KmsDriven = int(request.form.get('KmsDriven'))
    age = int(request.form.get('age'))
    OwnerType = int(request.form.get('OwnerType'))

    prediction = model.predict(pd.DataFrame([[bike_name,KmsDriven,age,brand,OwnerType]], columns=['bike_name','kms_driven','age','brand','Owner_Type']))
    return str(int(np.round(prediction[0])))


@app.route('/pop')
def pop():
    return render_template('pop.html')


if __name__ == "__main__":
    app.run(debug=True)