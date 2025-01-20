import streamlit as st
import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import os

# Paths to your files
h5_file_path = r"C:\Users\rajwn\Videos\Manus_Malaka\3DCardiac\data\ACDC\data\patient001_frame01.h5"
prediction_image_path = r"C:\Users\rajwn\Videos\Manus_Malaka\3DCardiac\prediction.png"
result_csv_path = r"C:\Users\rajwn\Videos\Manus_Malaka\3DCardiac\model\supervised\ACDC_BCP_7_labeled\unet_predictions\result.csv"

# Load the predicted image
@st.cache_data
def load_prediction_image(image_path):
    return Image.open(image_path)

# Load the .h5 file
@st.cache_data
def load_h5_file(file_path):
    with h5py.File(file_path, "r") as h5_file:
        data = {key: h5_file[key][:] for key in h5_file.keys()}
        return data

# Load evaluation metrics from CSV
@st.cache_data
def load_evaluation_metrics(csv_path):
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        st.error("Evaluation metrics CSV file not found!")
        return None

# Load data
data = load_h5_file(h5_file_path)
dataset_keys = list(data.keys())
predicted_image = load_prediction_image(prediction_image_path)
evaluation_metrics = load_evaluation_metrics(result_csv_path)

# Main title
st.title("Automatic Multi-Structure Cardiac MRI Segmentation")

# Section 1: Viewing images of .h5 and prediction
st.header("Viewing Images of .h5 and Prediction Based on Ground Truth")

# Subsection: Explore .h5 datasets
st.subheader("Explore H5 Data")
selected_dataset = st.selectbox("Select a Dataset", dataset_keys)
st.write(f"**Dataset Shape**: {data[selected_dataset].shape}")
st.write(f"**Data Type**: {data[selected_dataset].dtype}")

dataset = data[selected_dataset]

if len(dataset.shape) == 2:  # 2D image
    fig, ax = plt.subplots()
    ax.imshow(dataset, cmap="gray")
    ax.set_title(f"Dataset: {selected_dataset}")
    st.pyplot(fig)
elif len(dataset.shape) == 3:  # 3D volumetric data
    slice_idx = st.slider("Select Slice", 0, dataset.shape[0] - 1, dataset.shape[0] // 2)
    fig, ax = plt.subplots()
    ax.imshow(dataset[slice_idx, :, :], cmap="gray")
    ax.set_title(f"Dataset: {selected_dataset}, Slice: {slice_idx}")
    st.pyplot(fig)

# Subsection: Predicted Image
st.subheader("Predicted Image")
st.image(predicted_image, caption="Predicted Segmentation", use_column_width=True)

# Subsection: Segmentation Regions
st.subheader("Segmented Regions")
st.markdown("""
- **Background**: Purple
- **Left Ventricle**: Yellow
- **Right Ventricle**: Blue
- **Myocardium**: Green
""")

# Subsection: Evaluation Metrics
st.subheader("Evaluation Metrics")

if evaluation_metrics is not None:
    st.markdown("### Metrics Table")
    st.dataframe(evaluation_metrics)
    st.markdown("### Average Metrics")
    for end_type in ['ED', 'ES']:
        avg_row = evaluation_metrics[(evaluation_metrics['End_type'] == end_type) & (evaluation_metrics['Class'] == 'Average')]
        if not avg_row.empty:
            st.write(f"**{end_type} Average Metrics**")
            st.write(avg_row)
else:
    st.warning("No evaluation metrics available to display.")

# Subsection: Interactive Metric Selection
st.subheader("Interactive Metric Visualization")
if evaluation_metrics is not None:
    selected_class = st.selectbox("Select Class for Metrics", ["RV", "Myo", "LV", "Average"])
    metrics_to_plot = evaluation_metrics[evaluation_metrics['Class'] == selected_class]
    if not metrics_to_plot.empty:
        fig, ax = plt.subplots()
        ax.bar(metrics_to_plot['End_type'], metrics_to_plot['Dice'], label='Dice')
        ax.set_title(f"Dice Score for {selected_class}")
        ax.set_xlabel("End Type")
        ax.set_ylabel("Dice Score")
        st.pyplot(fig)
    else:
        st.warning(f"No metrics available for class {selected_class}.")
else:
    st.warning("No evaluation metrics available.")
