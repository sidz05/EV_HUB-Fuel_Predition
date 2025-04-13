import streamlit as st
import pandas as pd
import joblib
import os
from streamlit_lottie import st_lottie
import json
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="EV Energy Predictor", page_icon="🔋", layout="wide")

# Load model
model = joblib.load("best_model.pkl")

# Load animation
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Sidebar form
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/10467/10467994.png", width=100)
    st.title("🔋 Input Your Vehicle Specs")
    
    with st.form("prediction_form"):
        year = st.number_input("Year", min_value=2000, max_value=2030, value=2012)
        make = st.text_input("Make", value="MITSUBISHI")
        model_input = st.text_input("Model", value="i-MiEV")
        size = st.selectbox("Size", ["SUBCOMPACT", "COMPACT", "MID-SIZE", "FULL-SIZE"])
        type_ = st.selectbox("Type", ["A", "B", "C", "D"])
        kw = st.number_input("Power in kW", value=49.0)
        city_kwh = st.number_input("City kWh/100 km", value=16.9)
        hwy_kwh = st.number_input("Highway kWh/100 km", value=21.4)
        city_le = st.number_input("City Le/100 km", value=1.9)
        hwy_le = st.number_input("Highway Le/100 km", value=2.4)
        comb_le = st.number_input("Combined Le/100 km", value=2.1)
        g_km = st.number_input("g/km", value=0.0)
        km_range = st.number_input("Range in km", value=100.0)
        time_h = st.number_input("Charge Time (hours)", value=7.0)

        submit = st.form_submit_button("Predict")

# Main header
st.title("🔋 Vehicle Energy Consumption Predictor")
st.markdown("Predict **kWh/100 km** based on your vehicle specifications.")

# Lottie animation
lottie_animation = load_lottiefile("ev_animation.json")
st_lottie(lottie_animation, height=200, speed=1)

# Prediction result
if submit:
    user_input = pd.DataFrame([{
        'year': year,
        'make': make.upper().strip(),
        'model': model_input.strip(),
        'size': size.upper(),
        'type': type_.upper(),
        'kw': kw,
        'city_kwh/100_km': city_kwh,
        'hwy_kwh/100_km': hwy_kwh,
        'city_le/100_km': city_le,
        'hwy_le/100_km': hwy_le,
        'comb_le/100_km': comb_le,
        'g/km': g_km,
        'km': km_range,
        'time_h': time_h
    }])

    with st.spinner("🔍 Making prediction..."):
        prediction = model.predict(user_input)[0]

    st.success(f"✅ **Estimated Energy Consumption**: `{prediction:.2f} kWh/100 km`")

    result_df = pd.DataFrame([{"Predicted kWh/100 km": prediction}])
    st.download_button("⬇️ Download Prediction", result_df.to_csv(index=False), "prediction.csv", "text/csv")

# Model Comparison Plot (Dynamic)
st.markdown("---")
st.subheader("📊 Regression Model Comparison (Dynamic)")

# Model performance data
model_names = ["Linear", "Ridge", "Lasso", "Decision Tree"]
mae_scores = [0.0049, 0.0279, 0.1205, 0.3091]
rmse_scores = [0.0146, 0.0331, 0.1426, 0.6509]
r2_scores = [99.99, 99.95, 99.08, 80.84]  # Percent

# Plotting
fig, ax = plt.subplots(figsize=(10, 5))
bar_width = 0.25
x = range(len(model_names))

# Create grouped bars
ax.bar(x, mae_scores, width=bar_width, label='MAE', color='skyblue')
ax.bar([i + bar_width for i in x], rmse_scores, width=bar_width, label='RMSE', color='orange')
ax.bar([i + bar_width * 2 for i in x], r2_scores, width=bar_width, label='R² Score (%)', color='green')

# Formatting
ax.set_xlabel("Regression Models")
ax.set_ylabel("Scores")
ax.set_title("📊 Model Performance Comparison")
ax.set_xticks([i + bar_width for i in x])
ax.set_xticklabels(model_names)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Show chart
st.pyplot(fig)

# Print Accuracy (R² Score)
st.markdown("### ✅ Accuracy (R² Score %) of Each Model")
accuracy_df = pd.DataFrame({
    "Model": model_names,
    "Accuracy (R² %)": r2_scores
})
st.table(accuracy_df)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by **Siddhesh** | **Prerana** | **Rohit** ", unsafe_allow_html=True)
