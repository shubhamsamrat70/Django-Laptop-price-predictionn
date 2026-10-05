from django.shortcuts import render
from django.http import HttpResponse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder 
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from pandas import read_csv

# Create your views here.

def predict(request):
    df = pd.read_csv('predictionMl/static/csv/laptop_dataset_500.csv')
    X = df[["RAM", "Brand", "Processor", "Storage"]]
    y = df["Price"]
    col = ["Brand", "Processor"]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)

    process = ColumnTransformer(
        transformers=[
            ("laptop",OneHotEncoder(),col)
        ],
        remainder="passthrough"
    )

    model = Pipeline([
        ("preprocessor",process),
        ("regressor",DecisionTreeRegressor(random_state=42))
    ])


    model.fit(X_train,y_train)

    if request.method == "POST":
        ram = request.POST.get("RAM")
        Brand = request.POST.get("Brand")
        processor = request.POST.get("processor")
        storage = request.POST.get("storage")

        new_data = pd.DataFrame({
            "RAM":[ram],
            "Brand":[Brand],
            "Processor":[processor],
            "Storage":[storage]

        })
        y_pred = model.predict(X_test)

        accuracy = r2_score(y_test,y_pred)
        accuracy_percentage = accuracy * 100
        result = model.predict(new_data)
        price = result[0]
        print("Price", price)

        return render(request, 'predict.html',{
            "prediction":price,
            "accuracy":accuracy,
            "accuracy_percentage":accuracy_percentage,
            "ram":ram,
            "brand":Brand,
            "processor":processor,
            "storage":storage
        }
    )
    return render (request,"predict.html")    

        





