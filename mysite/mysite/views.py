from django.http import HttpResponse
from django.shortcuts import render
import joblib
import pandas as pd
import numpy as np
import sklearn
import pickle

def home(request):
    return render(request,'index.html')
    

def result(request):

    symptoms = request.POST["symptoms"]
    res = pred(symptoms)
    predic = res[0]
    print(predic)
    return render(request,'result.html',{'symptoms':symptoms,'pred':predic})

def pred(st):
    train = pd.read_csv('training.csv')
    model = joblib.load('model.joblib')
    df = train.iloc[[0]]
    for col in df.columns:
        df[col].values[:] = 0

    lst = st.split(',')

    for i in lst:
        for col in df.columns:
            if i==col:
                df.at[0, col] = 1

    df.drop('prognosis',inplace=True,axis=1)
    del df[df.columns[-1]] 
    return model.predict(df)

    