import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

INTERFACE = "gradio"   

# 1. Load dataset 
folder = os.path.dirname(os.path.abspath(__file__))
csv_files = [f for f in os.listdir(folder) if f.lower().endswith(".csv") and "electricity" in f.lower()]
if not csv_files:
    raise FileNotFoundError("No electricity CSV found in folder: " + folder)
path = os.path.join(folder, csv_files[0])
print("Using file:", path)
df = pd.read_csv(path)

# 2. Clean data
df = df.dropna(subset=["daily_consumption"])
for c in ["AWND", "PRCP", "TMAX", "TMIN"]:
    df[c] = df[c].fillna(df[c].median())

# 3. Features and target (predict daily consumption)
X = df[["AWND", "PRCP", "TMAX", "TMIN"]]
y = df["daily_consumption"]

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train linear regression
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# 7. Prediction function
def predict_usage(*values):
    try:
        input_data = pd.DataFrame([[float(v) for v in values]], columns=X.columns)
        usage = max(model.predict(input_data)[0], 0)
        return f"Predicted Daily Consumption: {usage:,.2f}"
    except Exception as e:
        return f"Prediction error: {str(e)}"

labels = ["Average Wind Speed (AWND)", "Precipitation (PRCP)",
          "Max Temperature (TMAX)", "Min Temperature (TMIN)"]

# 8. User interface
if INTERFACE == "gradio":
    import gradio as gr

    inputs = [gr.Number(label=lab, value=float(X[col].median()))
              for lab, col in zip(labels, X.columns)]

    demo = gr.Interface(
        fn=predict_usage,
        inputs=inputs,
        outputs=gr.Textbox(label="Prediction Result"),
        title="Electricity Consumption Prediction (Linear Regression)",
        description="Enter the weather details and click Submit to predict daily consumption."
    )
    demo.launch()

else:
    from flask import Flask, request

    app = Flask(__name__)

    @app.route("/", methods=["GET", "POST"])
    def home():
        result = ""
        if request.method == "POST":
            values = [request.form[col] for col in X.columns]
            result = "<h3>" + predict_usage(*values) + "</h3>"
        form = "".join(
            f"{lab}: <input name='{col}' type='number' step='any' "
            f"value='{X[col].median()}' required><br><br>"
            for lab, col in zip(labels, X.columns)
        )
        return ("<h2>Electricity Consumption Prediction (Linear Regression)</h2><form method='post'>"
                + form + "<button type='submit'>Predict</button></form>" + result)

    app.run()