import streamlit as st
import pandas as pd
import joblib

# Load the trained model and features
try:
    model_pipeline = joblib.load('ad_revenue_model.pkl')
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Error: The model file 'ad_revenue_model.pkl' was not found.")
    st.stop()

    # Title and description
st.title("YouTube Ad Revenue Predictor")
st.markdown("Enter video metrics to predict potential ad revenue.")

# Sidebar with user inputs
st.sidebar.header("Video Performance Metrics")
views = st.sidebar.number_input("Views", min_value=1, value=10000)
likes = st.sidebar.number_input("Likes", min_value=0, value=1000)
comments = st.sidebar.number_input("Comments", min_value=0, value=200)
watch_time_minutes = st.sidebar.number_input("Watch Time (minutes)", min_value=1.0, value=25000.0)
video_length_minutes = st.sidebar.number_input("Video Length (minutes)", min_value=0.1, value=10.0)
subscribers = st.sidebar.number_input("Subscribers", min_value=1, value=100000)

st.sidebar.header("Video Contextual Information")
category = st.sidebar.selectbox("Category", ['Gaming', 'Education', 'Entertainment', 'Music', 'Sports', 'News', 'Comedy'])
device = st.sidebar.selectbox("Device", ['TV', 'Tablet', 'Mobile', 'Desktop'])
country = st.sidebar.selectbox("Country", ['IN', 'UK', 'AU', 'CA', 'US'])

# Create a dataframe from user input
input_data = pd.DataFrame([{
    'views': views,
    'likes': likes,
    'comments': comments,
    'watch_time_minutes': watch_time_minutes,
    'video_length_minutes': video_length_minutes,
    'subscribers': subscribers,
    'category': category,
    'device': device,
    'country': country
}])

# Feature Engineering for the input data
input_data['engagement_rate'] = (input_data['likes'] + input_data['comments']) / input_data['views']

# Make prediction
if st.button("Predict Ad Revenue"):
    try:
        prediction = model_pipeline.predict(input_data)[0]
        st.success(f"### Predicted Ad Revenue: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed. Error: {e}")

 # Basic visualizations
st.markdown("---")
st.subheader("Model Insights")
st.write("This section could show feature importance or other visualizations from the model.")
# Example visualization (you'd need to adapt this based on your model)
from sklearn.ensemble import RandomForestRegressor


# Dynamically find the regressor step with feature_importances_ or coef_
reg = model_pipeline.named_steps['regressor']

if hasattr(reg, "coef_"):
    pre = model_pipeline.named_steps['preprocessor']
    cat_names = pre.named_transformers_['cat'].get_feature_names_out(['category','device','country'])
    num_names = ['views','likes','comments','watch_time_minutes','video_length_minutes','subscribers','engagement_rate']
    feature_names = list(num_names) + list(cat_names)

    coefs = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': reg.coef_
    }).sort_values('Coefficient', ascending=False)

    st.bar_chart(coefs.set_index('Feature'))
    st.dataframe(coefs.head(30))
# ...existing code...


    
          
