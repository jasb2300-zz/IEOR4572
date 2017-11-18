# Emily Jin and Jorge Solis
# May 3rd, 2017
# Professor Hardeep Johar
# Data Analytics for Operations Research

#necessary libraries
from flask import Flask, session, render_template, request, make_response, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import StringField
from wtforms.validators import DataRequired
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from scipy import stats
from textwrap import wrap
import matplotlib.pyplot as plt
import pylab
import numpy as np
import pandas as pd
import plotly

df = pd.read_csv('out.csv',encoding='latin1')
df = df[df['Life expectancy at birth (years)'] != '_']


app = Flask(__name__)

app.secret_key = 'CSDGvasd0923u4jgd_gfdafgjnp398jsk'

@app.route('/')
def home():
    session['data_loaded'] = True
    return render_template('home.html')

@app.route('/compute', methods=['GET'])
def compute():
    #parse url request
    at = request.args.get('attribute')
    at = at.replace('+',' ')
    at = at.replace('%28','(')
    at = at.replace('%29',')')
    attr = list(df[at])
    life = list(df['Life expectancy at birth (years)'])
    df_test = pd.DataFrame({'Attribute':attr, 'Life Expectancy':life})

    #check for missing data
    missing = 0
    for i in attr:
        if i == '_':
            missing = 1

    if missing == 1:
        df_test = df_test[df_test['Attribute'] != '_']

    #statistical analysis and ML regression
    train, test = train_test_split(df_test, test_size = 0.3)
    x_train = pd.to_numeric(train.iloc[0:,0])
    y_train = pd.to_numeric(train.iloc[0:,1])
    x_test = pd.to_numeric(test.iloc[0:,0])
    y_test = pd.to_numeric(test.iloc[0:,1])

    model = linear_model.LinearRegression()
    model.fit(x_train.values.reshape(-1,1),y_train.values.reshape(-1,1))

    #find in and out sample mse
    training_predictions = model.predict(x_train.values.reshape(-1,1))
    mse_in = np.mean((training_predictions - y_train.values.reshape(-1,1))**2)
    test_predictions = model.predict(x_test.values.reshape(-1,1))
    mse_out = np.mean((test_predictions - y_test.values.reshape(-1,1))**2)

    #for overall data
    x = [float(i) for i in df_test['Attribute']]
    y = [float(i) for i in df_test['Life Expectancy']]

    #assign values for display
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    r2 = r_value**2
    stat = [mse_in,mse_out,r_value,r2]

    #write png file and graph scatterplot with regression line
    fig = plt.figure()
    plt.scatter(x,y)
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
    plt.ylabel('Expected Lifespan')
    plt.xlabel('\n'.join(wrap(at,60)))
    plt.subplots_adjust(bottom=.2)
    plt.savefig('static/plot.png')
    plt.close()

    #Convert CSV file into Pandas DataFrame
    #Prepare values for plot function
    country = list(df['Member State'])
    code = list(df['Code'])
    attr = list(df[at])
    life = list(df['Life expectancy at birth (years)'])
    df_map = pd.DataFrame({'Country':country, 'Code':code, 'Attribute':attr})

    #prepare data for mapping
    data = [ dict( type = 'choropleth',locations = df_map['Code'],
        z = df_map['Attribute'],
        text = df_map['Country'],
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
                [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
                ) ),
            colorbar = dict(
                autotick = False,
                title = 'Values'),
            ) ]
    #Configure layout for plot function
    layout = dict(
            title = at,
            geo = dict(
                showframe = False,
                showcoastlines = False,
                projection = dict(
                    type = 'Mercator'
                    )
                )
            )
    #Plot correlation mapping
    fig = dict( data=data, layout=layout)
    plotly.offline.plot( fig, validate=False, filename='static/map.html')

    return render_template('compute.html', attributes=at, values=stat)

#clear cache
@app.after_request
def add_header(response):
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

if __name__ == '__main__':
    app.run()
