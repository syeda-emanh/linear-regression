import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

INTERFACE = "gradio"    

# 1. Load dataset 
folder = os.path.dirname(os.path.abspath(__file__))
csv_files = [f for f in os.listdir(folder) if f.lower().endswith(".csv") and "car" in f.lower()]
if not csv_files:
    raise FileNotFoundError("No car CSV found in folder: " + folder)
path = os.path.join(folder, csv_files[0])
print("Using file:", path)
df = pd.read_csv(path)

# 2. Features and target (predict selling price)
# Convert text columns to numbers (one-hot encoding)
X = pd.get_dummies(df[["year", "km_driven", "fuel", "seller_type", "transmission", "owner"]],
                   drop_first=True)
y = df["selling_price"]

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train linear regression
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# 6. Prediction function
fuels = sorted(df["fuel"].unique())
sellers = sorted(df["seller_type"].unique())
transmissions = sorted(df["transmission"].unique())
owners = sorted(df["owner"].unique())

def predict_car(year, km_driven, fuel, seller_type, transmission, owner):
    try:
        row = pd.DataFrame([{"year": float(year), "km_driven": float(km_driven),
                             "fuel": fuel, "seller_type": seller_type,
                             "transmission": transmission, "owner": owner}])
        # same encoding as training data
        row = pd.get_dummies(row).reindex(columns=X.columns, fill_value=0)
        price = max(model.predict(row.astype(float))[0], 0)
        return f"Predicted Selling Price: {price:,.0f}"
    except Exception as e:
        return f"Prediction error: {str(e)}"

# 7. User interface
if INTERFACE == "gradio":
    import gradio as gr

    inputs = [
        gr.Number(label="Year", value=2014),
        gr.Number(label="Kilometers Driven", value=60000),
        gr.Dropdown(fuels, value="Petrol", label="Fuel"),
        gr.Dropdown(sellers, value="Individual", label="Seller Type"),
        gr.Dropdown(transmissions, value="Manual", label="Transmission"),
        gr.Dropdown(owners, value="First Owner", label="Owner"),
    ]

    demo = gr.Interface(
        fn=predict_car,
        inputs=inputs,
        outputs=gr.Textbox(label="Prediction Result"),
        title="Used Car Price Prediction (Linear Regression)",
        description="Enter the car details and click Submit to predict the selling price."
    )
    demo.launch()

else:
    from flask import Flask, request

    app = Flask(__name__)

    def select(name, label, options, default):
        opts = "".join(f"<option {'selected' if o == default else ''}>{o}</option>" for o in options)
        return f"{label}: <select name='{name}'>{opts}</select><br><br>"

    @app.route("/", methods=["GET", "POST"])
    def home():
        result = ""
        if request.method == "POST":
            f = request.form
            result = "<h3>" + predict_car(f["year"], f["km_driven"], f["fuel"],
                                          f["seller_type"], f["transmission"], f["owner"]) + "</h3>"
        form = ("Year: <input name='year' type='number' value='2014' required><br><br>"
                "Kilometers Driven: <input name='km_driven' type='number' value='60000' required><br><br>"
                + select("fuel", "Fuel", fuels, "Petrol")
                + select("seller_type", "Seller Type", sellers, "Individual")
                + select("transmission", "Transmission", transmissions, "Manual")
                + select("owner", "Owner", owners, "First Owner"))
        return ("<h2>Used Car Price Prediction (Linear Regression)</h2><form method='post'>"
                + form + "<button type='submit'>Predict</button></form>" + result)

    app.run()