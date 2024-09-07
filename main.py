import os
import subprocess
import sys

# Install matplotlib if it's not already installed
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt


import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


# Set Seaborn style
sns.set_style("whitegrid")

# Define custom color palettes
custom_palette = {
    'Scatter Plot': sns.color_palette("husl"),
    'Bar Graph': sns.color_palette("pastel"),
    'Histogram': sns.color_palette("dark")
}

# Title and Image
st.title("Disease Prognosis and Analysis")
st.image("skin.png", width=500)

# Display the dataset link
st.markdown("[Download the dataset from Kaggle](https://www.kaggle.com/datasets/omersedawei/cvd-cleaned)")

# Load dataset
dataset_path = r"C:\ITBIN-2110-0069\Disease-prognosis-app\CVD_cleaned.csv"  # Full path to the CSV 
data = pd.read_csv(dataset_path)
st.write(f"Dataset shape: {data.shape}")

# Sidebar menu with additional filters
st.sidebar.title("Navigation")
menu_option = st.sidebar.radio("Select an option", ["Home", "Prediction Details"])

if menu_option == "Home":
    st.sidebar.subheader("Home Section")
    st.image("predictimg.png", width=550)

    st.header("Filters")
    # Add filters for data
    min_height = st.sidebar.slider("Minimum Height (cm)", min_value=0, max_value=300, value=0)
    max_height = st.sidebar.slider("Maximum Height (cm)", min_value=0, max_value=300, value=300)
    min_weight = st.sidebar.slider("Minimum Weight (kg)", min_value=0, max_value=300, value=0)
    max_weight = st.sidebar.slider("Maximum Weight (kg)", min_value=0, max_value=300, value=300)
    
    filtered_data = data[(data["Height_(cm)"] >= min_height) & (data["Height_(cm)"] <= max_height) &
                         (data["Weight_(kg)"] >= min_weight) & (data["Weight_(kg)"] <= max_weight)]

    st.header("Tabular Data")
    if st.checkbox("Show Tabular Data"):
        st.write(filtered_data.head(150))

    st.header("Statistical Summary")
    if st.checkbox("Show Statistics"):
        st.write(filtered_data.describe())

    st.header("Correlation Heatmap")
    if st.checkbox("Show Correlation Heatmap"):
        numeric_data = filtered_data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Heatmap", fontsize=16)
            st.pyplot(fig)
        else:
            st.error("No numeric data available to compute correlation.")

    st.header("Visualizations")
    graph = st.selectbox("Choose the type of graph", ["Scatter Plot", "Bar Graph", "Histogram"])

    if graph == "Scatter Plot":
        x_col = st.selectbox("Select x-axis column", filtered_data.select_dtypes(include=[np.number]).columns)
        y_col = st.selectbox("Select y-axis column", filtered_data.select_dtypes(include=[np.number]).columns)
        hue_col = st.selectbox("Select hue column (optional)", ["None"] + list(filtered_data.select_dtypes(include=[object]).columns))
        hue_col = None if hue_col == "None" else hue_col

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.scatterplot(data=filtered_data, x=x_col, y=y_col, hue=hue_col, palette=custom_palette['Scatter Plot'], ax=ax)
        ax.set_title(f"Scatter Plot of {x_col} vs {y_col}")
        st.pyplot(fig)

    elif graph == "Bar Graph":
        column_to_plot = st.selectbox("Select column to plot", filtered_data.select_dtypes(include=[object]).columns)
        fig, ax = plt.subplots(figsize=(12, 6))
        counts = filtered_data[column_to_plot].value_counts().reset_index()
        counts.columns = [column_to_plot, 'Count']

        sns.barplot(x=column_to_plot, y='Count', data=counts, palette=custom_palette['Bar Graph'], ax=ax)
        ax.set_title(f'Count of Occurrences for {column_to_plot}')
        st.pyplot(fig)

    elif graph == "Histogram":
        column_to_plot = st.selectbox("Select column to plot", filtered_data.select_dtypes(include=[np.number]).columns)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(filtered_data[column_to_plot], kde=True, color=custom_palette['Histogram'][0], ax=ax)
        ax.set_title(f'Histogram of {column_to_plot}')
        ax.set_xlabel(column_to_plot)
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

    # Media display section
    st.header("Upload your media for prediction:")
    media_file = st.file_uploader("Upload an image, video, or audio file", type=["png", "jpg", "jpeg", "mp4", "mp3", "wav"])
    
    if media_file is not None:
        file_details = {"FileName": media_file.name, "FileType": media_file.type}
        st.write(file_details)
        
        if media_file.type.startswith('image'):
            st.image(media_file)
        elif media_file.type.startswith('video'):
            st.video(media_file)
        elif media_file.type.startswith('audio'):
            st.audio(media_file)

    # Provide option to download filtered data
    st.download_button(
        label="Download Filtered Data",
        data=filtered_data.to_csv(index=False).encode('utf-8'),
        file_name='filtered_data.csv',
        mime='text/csv'
    )

elif menu_option == "Prediction Details":
    st.sidebar.subheader("Prediction Section")
    st.title("Patient Weight Prediction")

    # Prepare the data
    X = np.array(data["Height_(cm)"]).reshape(-1, 1)
    y = np.array(data["Weight_(kg)"]).reshape(-1, 1)
    
    # Create and train the model
    lregression = LinearRegression()
    lregression.fit(X, y)
    
    # Create the number input for height
    value = st.number_input("Enter Height (cm)", min_value=1, max_value=300, value=150, step=1)
    
    # Reshape the input value and make prediction
    value = np.array(value).reshape(1, -1)
    prediction = lregression.predict(value)[0][0]
    
    # Display the result
    st.write(f"Predicted Weight for a Height of {value[0][0]} cm: {prediction:.2f} kg")
