from flask import Flask, request, render_template
import pandas as pd
import pickle
import numpy as np


app = Flask("WallStreet and Your Street")

model = pickle.load(open("model.pkl", "rb"))

if __name__ == "main":
    app.run(debug = True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        stockprice = request.form.get("stockprice")
        stock(stockprice)
        X = pd.DataFrame([stockprice])
        prediction = model.predict(np.array(X).reshape(-1, 1))[0]
    else:
        prediction = ""
    return render_template('chart.html', output = prediction)

@app.route('/stock')
def stock(stockprice):
    return stockprice

@app.route('/pred')
def pred(stockprice):
    X = pd.DataFrame([stockprice])
    prediction = model.predict(np.array(X).reshape(-1, 1))[0]
    return prediction

@app.route('/team')
def team():    
    return render_template('team.html')

@app.route('/key')
def key():    
    return render_template('key.html')

@app.route('/about')
def about():    
    return render_template('about.html')

