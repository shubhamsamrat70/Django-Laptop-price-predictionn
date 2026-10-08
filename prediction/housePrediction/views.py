from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score


def housePredict(request):

    df = pd.read_csv(
        'predictionMl/static/csv/HouseP.csv',
        sep='\t'
    )

    print(df.columns)

    X = df[
        [
            "State",
            "Pin Code",
            "Sub District",
            "seller_type",
            "layout_type",
            "furnish_type",
            "floors",
            "bathrooms",
            "bedroom",
            "Waterfront_View",
            "grade"
        ]
    ]

    y = df["price"]

    col = ["State", "Sub District"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=25,
        random_state=42
    )

    process = ColumnTransformer(
        transformers=[
            ("House", OneHotEncoder(handle_unknown="ignore"), col)
        ],
        remainder="passthrough"
    )

    model = Pipeline([
        ("preprocessor", process),
        ("regressor", DecisionTreeRegressor(random_state=42))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = r2_score(y_test, y_pred)
    accuracy_percentage = accuracy * 100

    if request.method == "POST":

        state = request.POST.get("State")
        pin_code = int(request.POST.get("Pin Code"))
        sub_district = request.POST.get("Sub District")
        seller_type = int(request.POST.get("seller_type"))
        layout_type = int(request.POST.get("layout_type"))
        furnish_type = int(request.POST.get("furnish_type"))
        floors = int(request.POST.get("floors"))
        bathrooms = int(request.POST.get("bathrooms"))
        bedroom = int(request.POST.get("bedroom"))
        waterfront_view = int(request.POST.get("Waterfront_View"))
        grade = int(request.POST.get("grade"))

        new_data = pd.DataFrame({
            "State": [state],
            "Pin Code": [pin_code],
            "Sub District": [sub_district],
            "seller_type": [seller_type],
            "layout_type": [layout_type],
            "furnish_type": [furnish_type],
            "floors": [floors],
            "bathrooms": [bathrooms],
            "bedroom": [bedroom],
            "Waterfront_View": [waterfront_view],
            "grade": [grade]
        })

        result = model.predict(new_data)

        price = result[0]

        print("Prediction Price:" \
        "",price)

        return render(
            request,
            "housePredict.html",
            {
                "predictions": price,
                "accuracy": accuracy,
                "accuracy_percentage": accuracy_percentage,

                "State": state,
                "Pin Code": pin_code,
                "Sub District": sub_district,
                "seller_type": seller_type,
                "layout_type": layout_type,
                "furnish_type": furnish_type,
                "floors": floors,
                "bathrooms": bathrooms,
                "bedroom": bedroom,
                "Waterfront_View": waterfront_view,
                "grade": grade
            }
        )

    return render(
        request,
        "housePredict.html",
        {
            "accuracy": accuracy,
            "accuracy_percentage": accuracy_percentage
        }
    )