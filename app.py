import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv("final_dataset.csv")
df = df.drop(columns=["Unnamed: 0"], errors='ignore')

# Streamlit app
st.set_page_config(page_title="House Price Predictor", layout="wide")
st.title("🏠 House Price Prediction")

# Show dataset preview
if st.checkbox("Show raw dataset"):
    st.write(df.head())

# Visualizations
st.subheader("📊 Data Visualization")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Correlation Heatmap**")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

with col2:
    st.markdown("**Price Distribution**")
    fig2, ax2 = plt.subplots()
    sns.histplot(df['price'], kde=True, ax=ax2)
    st.pyplot(fig2)

# Feature selection
X = df.drop("price", axis=1)
y = df["price"]

# Encode categorical if needed
X = pd.get_dummies(X, drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection
st.sidebar.header("⚙️ Model & Input")
model_name = st.sidebar.selectbox("Choose ML Model", ["Linear Regression", "K-Nearest Neighbors"])
if model_name == "K-Nearest Neighbors":
    k = st.sidebar.slider("K value", 1, 20, 5)
    model = KNeighborsRegressor(n_neighbors=k)
else:
    model = LinearRegression()

# Sidebar user input
st.sidebar.header("🏠 Enter House Features")

beds = st.sidebar.slider("Bedrooms", 0, 10, 3)
baths = st.sidebar.slider("Bathrooms", 0.0, 10.0, 2.0, 0.5)
size = st.sidebar.slider("Size (sq ft)", 200, 10000, 1500)

if "zip_code" in df.columns:
    zip_code = st.sidebar.selectbox("ZIP Code", sorted(df["zip_code"].unique()))
    input_df = pd.DataFrame({
        "beds": [beds],
        "baths": [baths],
        "size": [size],
        "zip_code": [zip_code]
    })
    input_df = pd.get_dummies(input_df)
    # Align input to training data
    input_df = input_df.reindex(columns=X.columns, fill_value=0)
else:
    input_df = pd.DataFrame({
        "beds": [beds],
        "baths": [baths],
        "size": [size]
    })

# Train model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Performance
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.subheader("📈 Model Performance")
st.markdown(f"**R² Score:** `{r2:.3f}`")
st.markdown(f"**RMSE:** `${rmse:,.2f}`")

# Predict user input
if st.sidebar.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.sidebar.success(f"💰 Predicted Price: ${prediction:,.2f}")

# Actual vs Predicted
st.subheader("📉 Actual vs Predicted Prices")
fig3, ax3 = plt.subplots()
ax3.scatter(y_test, y_pred, alpha=0.6)
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax3.set_xlabel("Actual Price")
ax3.set_ylabel("Predicted Price")
st.pyplot(fig3)
