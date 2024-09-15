import streamlit as st
import numpy as np
import pandas as pd
import time
# from streamlit import caching
from PIL import Image
from preprocessing import *
from models import *
from dataLoader import *
import torch
import os
import matplotlib.pyplot as plt
import gif_your_nifti.core as gif2nif
import shutil
import time
from visualization import *
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go

st.set_page_config(
    page_title="SAD-Unet",
    page_icon="./vc.png"
)


st.title('3D Brain MRI Segmentation App')

st.markdown("***")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# st.subheader('Upload the MRI scan of the brain')
# option = st.radio('',('Single MRI scan', 'Multiple MRI scans'))
st.write('Runing on:', device)

model_path = "./saved_model/best_model_SAD_UNet.pt"

@st.cache_resource
def load_model():
    model = SAD_UNet(in_channels=4, num_classes=4)  # Assuming SAD_UNet is defined elsewhere
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device), weights_only=True))
    model.eval()
    return model

# Load the model only once
model = load_model()
# if option == 'Multiple MRI scans':
st.subheader('Upload the MRI scans of the brain')
uploaded_files = st.file_uploader(' ',accept_multiple_files = True, type=["nii", "nii.gz"])

if 'file_info_list' not in st.session_state:
    st.session_state['file_info_list'] = {}
    st.session_state['prev_uploaded_files'] = []

file_info_list = {}
file_names = [uploaded_file.name for uploaded_file in uploaded_files] if uploaded_files else []

if uploaded_files:
    if uploaded_files and file_names != st.session_state['prev_uploaded_files']:
        for uploaded_file in uploaded_files:
            save_path = uploaded_file.name

            if "t1c" in save_path:
                file_info_list["t1c"] = save_path
            elif "t1n" in save_path:
                file_info_list["t1n"] = save_path
            elif "t2f" in save_path:
                file_info_list["t2f"] = save_path
            elif "t2w" in save_path:
                file_info_list["t2w"] = save_path

            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        st.session_state['file_info_list'] = file_info_list
    
    sagittal_file = 'sagittal.gif'
    coronal = 'coronal.gif'
    axial = 'axial.gif'

    file_info_list = st.session_state['file_info_list']
    

    if uploaded_files and file_names != st.session_state['prev_uploaded_files']:
        create_rotated_sagittal_gif_from_nifti(file_info_list.get("t1c"), sagittal_file)
        create_rotated_coronal_gif_from_nifti(file_info_list.get("t1c"), coronal)
        create_rotated_axial_gif_from_nifti(file_info_list.get("t1c"), axial)

    image_paths = ["./sagittal.gif", "./coronal.gif", "./axial.gif"]

    

    # Create three columns
    col1, col2, col3 = st.columns(3)

    # Display images in each column
    with col1:
        st.image(image_paths[0], caption="sagittal angle", use_column_width=True)

    with col2:
        st.image(image_paths[1], caption="coronal angle", use_column_width=True)

    with col3:
        st.image(image_paths[2], caption="axial angle", use_column_width=True)


    tab1, tab2, tab3 = st.tabs(["Coronal", "Axial", "Sagittal"] )

    with tab1 : 

        slices = st.slider("Choose number of Slice to display", min_value=10, max_value=144, value=80, step=1, key="coronal_slider")

        n_slice = slices
        image_t1n = nib.load(file_info_list.get("t1n")).get_fdata()
        image_t1c = nib.load(file_info_list.get("t1c")).get_fdata()
        image_t2w = nib.load(file_info_list.get("t2w")).get_fdata()
        image_t2f = nib.load(file_info_list.get("t2f")).get_fdata()

        # Extract slices
        test_image_t1n = image_t1n[:, :, n_slice]
        test_image_t1c = image_t1c[:, :, n_slice]
        test_image_t2w = image_t2w[:, :, n_slice]
        test_image_flair = image_t2f[:, :, n_slice]

        fig = sp.make_subplots(rows=2, cols=2, subplot_titles=("Image T1n", "Image T1c", "Image T2w", "Image Flair"))

    # Create subplots with Plotly
        fig.add_trace(go.Heatmap(z=test_image_t1n, colorscale='gray', showscale=False), row=1, col=1)
        fig.add_trace(go.Heatmap(z=test_image_t1c, colorscale='gray', showscale=False), row=1, col=2)
        fig.add_trace(go.Heatmap(z=test_image_t2w, colorscale='gray', showscale=False), row=2, col=1)
        fig.add_trace(go.Heatmap(z=test_image_flair, colorscale='gray', showscale=False), row=2, col=2)

        # Update layout to match dark theme
        fig.update_layout(
            height=700, 
            width=800, 
            title_text="Image and Mask Visualization",
            title_font_color="white",
            paper_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background to match site
            plot_bgcolor='rgba(0, 0, 0, 0)',   # Transparent plot area
            font=dict(color="white"),           # White font for better contrast
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig)

    with tab2 : 

        slices = st.slider("Choose number of Slice to display", min_value=38, max_value=211, value=80, step=1, key="Axial_slider")

        n_slice = slices
        image_t1n = nib.load(file_info_list.get("t1n")).get_fdata()
        image_t1c = nib.load(file_info_list.get("t1c")).get_fdata()
        image_t2w = nib.load(file_info_list.get("t2w")).get_fdata()
        image_t2f = nib.load(file_info_list.get("t2f")).get_fdata()

        # Extract slices
        test_image_t1n = image_t1n[:, n_slice, :].T
        test_image_t1c = image_t1c[:, n_slice, :].T
        test_image_t2w = image_t2w[:, n_slice, :].T
        test_image_flair = image_t2f[:, n_slice, :].T

        fig = sp.make_subplots(rows=2, cols=2, subplot_titles=("Image T1n", "Image T1c", "Image T2w", "Image Flair"))

    # Create subplots with Plotly
        fig.add_trace(go.Heatmap(z=test_image_t1n, colorscale='gray', showscale=False), row=1, col=1)
        fig.add_trace(go.Heatmap(z=test_image_t1c, colorscale='gray', showscale=False), row=1, col=2)
        fig.add_trace(go.Heatmap(z=test_image_t2w, colorscale='gray', showscale=False), row=2, col=1)
        fig.add_trace(go.Heatmap(z=test_image_flair, colorscale='gray', showscale=False), row=2, col=2)

        # Update layout to match dark theme
        fig.update_layout(
            height=700, 
            width=800, 
            title_text="Image and Mask Visualization",
            title_font_color="white",
            paper_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background to match site
            plot_bgcolor='rgba(0, 0, 0, 0)',   # Transparent plot area
            font=dict(color="white"),           # White font for better contrast
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig)

    with tab3 : 

        slices = st.slider("Choose number of Slice to display", min_value=54, max_value=186, value=80, step=1, key="Sagittal_slider")

        n_slice = slices
        image_t1n = nib.load(file_info_list.get("t1n")).get_fdata()
        image_t1c = nib.load(file_info_list.get("t1c")).get_fdata()
        image_t2w = nib.load(file_info_list.get("t2w")).get_fdata()
        image_t2f = nib.load(file_info_list.get("t2f")).get_fdata()

        # Extract slices
        test_image_t1n = image_t1n[n_slice, :, :].T
        test_image_t1c = image_t1c[n_slice, :, :].T
        test_image_t2w = image_t2w[n_slice, :, :].T
        test_image_flair = image_t2f[n_slice, :, :].T

        fig = sp.make_subplots(rows=2, cols=2, subplot_titles=("Image T1n", "Image T1c", "Image T2w", "Image Flair"))

    # Create subplots with Plotly
        fig.add_trace(go.Heatmap(z=test_image_t1n, colorscale='gray', showscale=False), row=1, col=1)
        fig.add_trace(go.Heatmap(z=test_image_t1c, colorscale='gray', showscale=False), row=1, col=2)
        fig.add_trace(go.Heatmap(z=test_image_t2w, colorscale='gray', showscale=False), row=2, col=1)
        fig.add_trace(go.Heatmap(z=test_image_flair, colorscale='gray', showscale=False), row=2, col=2)

        # Update layout to match dark theme
        fig.update_layout(
            height=700, 
            width=800, 
            title_text="Image and Mask Visualization",
            title_font_color="white",
            paper_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background to match site
            plot_bgcolor='rgba(0, 0, 0, 0)',   # Transparent plot area
            font=dict(color="white"),           # White font for better contrast
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig)




    st.subheader('Upload the Ground Truth Mask for Comparison')
    gt_file = st.file_uploader(' ', accept_multiple_files=False, type=["nii", "nii.gz"])

    if gt_file:
        mask_name = gt_file.name
        with open(mask_name, "wb") as f:
                f.write(gt_file.getbuffer())
        imageww = nib.load(mask_name).get_fdata()
        # st.write('You selected:', np.unique(imageww))

        preprocessing_mask = mask_preprocessing(mask_name)
        mask_path = f'./{preprocessing_mask}.npy'
        mask_gt = np.load(mask_path)

    if uploaded_files and file_names != st.session_state['prev_uploaded_files']:
        preprocessing_img = nnunet_to_npy(file_info_list)

        # Assuming you have a model defined elsewhere
        model = model.to(device)
        img_path = f'./{preprocessing_img}.npy'
        image = np.load(img_path)

        # Load only one image for evaluation
        image = torch.tensor(image).unsqueeze(0).to(device)  # Add batch dimension and move to GPU

        # Prediction
        with torch.no_grad():
            output, *_ = model(image)
            prediction = output.cpu().numpy()

        # Save the prediction
        np.save(f"./predictions/{preprocessing_img}.npy", prediction)
        st.session_state['preprocessing_img'] = preprocessing_img

    teesrt = st.session_state['preprocessing_img']
    img_path = f'./{teesrt}.npy'
    val_pred = np.load(f"./predictions/{teesrt}.npy")
    val_pred = val_pred.argmax(axis=1)

    tab4, tab5, tab6 = st.tabs(["Coronal", "Axial", "Sagittal"])

    with tab4:
        n_slice = st.slider("Choose number of Slice to display", min_value=1, max_value=127, value=80, step=1, key="slider1")

        # Extract the slices for plotting
        pred_slice = val_pred[0, :, :, n_slice]
        imagew = np.load(img_path)
        t1c_slice = imagew[1, :, :, :]
        t1c_slice = t1c_slice[:, :, n_slice]

        if np.all(pred_slice == 0):
            st.markdown('<p style="color:green; font-weight:bold;">No tumor detected in this slice.</p>', unsafe_allow_html=True)
        else:
            # Mask the prediction where the value is 0 (background)
            combined_img = np.ma.masked_where(pred_slice == 0, pred_slice)

            # Custom colormap for the prediction mask
            custom_colorscale = [
                [0.0, 'rgba(0,0,0,0)'],  # Fully transparent where there is no tumor (background)
                [0.33, 'green'],  # Start with green
                [0.66, 'yellow'],  # Then yellow
                [1.0, 'red']       # End with red
            ]

            # Custom colormap for ground truth overlay
            custom_colorscale_gt = [
                [0.0, 'rgba(0,0,0,0)'],  # Fully transparent where there is no tumor (background)
                [0.33, 'green'],  # Start with green
                [0.66, 'yellow'],  # Then yellow
                [1.0, 'red']       # End with red
            ]

            # Custom colormap for the background image
            custom_colorscale2 = [
                [0.0, 'black'],   # Black where there is no tumor (background)
                [0.33, 'green'],  # Start with green
                [0.66, 'yellow'], # Then yellow
                [1.0, 'red']      # End with red
            ]

            # Create subplots with 1 row and 2 columns
            figs = sp.make_subplots(rows=1, cols=2, subplot_titles=("Prediction", "T1c + Prediction Overlay"))

            # Add the prediction heatmap to the first subplot using the custom colorscale
            figs.add_trace(go.Heatmap(z=pred_slice, colorscale=custom_colorscale2, showscale=False), row=1, col=1)

            # Add grayscale MRI slice to the second subplot
            figs.add_trace(go.Heatmap(
                z=t1c_slice,
                colorscale='gray',
                showscale=False,
                hoverinfo='skip'  # Disable hover info for the background image
            ), row=1, col=2)

            # Add overlay with the prediction mask using the custom colormap to the second subplot
            figs.add_trace(go.Heatmap(
                z=combined_img,
                colorscale=custom_colorscale,
                showscale=False,
                opacity=0.6,  # Set transparency
                hoverinfo='skip'  # Disable hover info for the overlay
            ), row=1, col=2)

            # Update layout
            figs.update_layout(
                height=400,
                width=700,
                title_text="Prediction and T1c + Prediction Overlay",
                title_font_color="white",
                paper_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
                plot_bgcolor='rgba(0, 0, 0, 0)',   # Transparent plot area
                font=dict(color="white")           # White font for contrast
            )

            # Display the combined plot
            st.plotly_chart(figs)

        if uploaded_files and file_names == st.session_state['prev_uploaded_files']:
            # Conditional row for ground truth comparison
            if gt_file:
                imageww = nib.load(gt_file.name).get_fdata()
                mask_slicec = imageww[56:184, 56:184, 13:141]
                # Extract the ground truth slice
                mask_slice = mask_slicec[:, :, n_slice]  # Adjust this based on ground truth array dimensions
                # mask_slice = mask_gt[0, :, :, n_slice]  # Adjust this based on ground truth array dimensions
                # st.write('You selected:', np.unique(mask_gt))

                if np.all(mask_slice == 0):
                    st.markdown('<p style="color:green; font-weight:bold;">The ground truth indicates the absence of a tumor.</p>', unsafe_allow_html=True)
                else:

                    custom_colorscale2 = [
                        [0.0, 'black'],   # Black where there is no tumor (background)
                        [0.33, 'green'],  # Start with green
                        [0.66, 'yellow'], # Then yellow
                        [1.0, 'red']      # End with red
                    ]

                    custom_colorscale_gt = [
                        [0.0, 'rgba(0,0,0,0)'],  # Fully transparent where there is no tumor (background)
                        [0.33, 'green'],  # Start with green
                        [0.66, 'yellow'],  # Then yellow
                        [1.0, 'red']       # End with red
                    ]
                    # Create a new row for ground truth comparison
                    figs_gt = sp.make_subplots(rows=1, cols=2, subplot_titles=("Ground Truth", "T1c + Ground Truth Overlay"))

                    # Add the ground truth mask to the first subplot using the custom colormap
                    figs_gt.add_trace(go.Heatmap(z=mask_slice, colorscale=custom_colorscale2, showscale=False), row=1, col=1)

                    # Add overlay with the ground truth mask using the custom colormap to the second subplot
                    figs_gt.add_trace(go.Heatmap(
                        z=t1c_slice,
                        colorscale='gray',
                        showscale=False,
                        hoverinfo='skip'
                    ), row=1, col=2)

                    figs_gt.add_trace(go.Heatmap(
                        z=np.ma.masked_where(mask_slice == 0, mask_slice),  # Mask the ground truth
                        colorscale=custom_colorscale_gt,
                        showscale=False,
                        opacity=0.6,  # Set transparency
                        hoverinfo='skip'
                    ), row=1, col=2)

                    # Update layout
                    figs_gt.update_layout(
                        height=400,
                        width=700,
                        title_text="Ground Truth and T1c + Ground Truth Overlay",
                        title_font_color="white",
                        paper_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
                        plot_bgcolor='rgba(0, 0, 0, 0)',   # Transparent plot area
                        font=dict(color="white")           # White font for contrast
                    )

                    # Display the ground truth plot
                    st.plotly_chart(figs_gt)


    with tab5 : 

        n_slice = st.slider("Choose number of Slice to display", min_value=1, max_value=127, value=80, step=1, key="slider2")

        # Extract the slices for plotting
        pred_slice = val_pred[0, :, n_slice, :].T
        imagew = np.load(img_path)
        t1c_slice = imagew[1, :, :, :] 
        t1c_slice = t1c_slice[:, n_slice, :].T

        if np.all(pred_slice == 0):
            st.markdown('<p style="color:green; font-weight:bold;">No tumor detected in this slice.</p>', unsafe_allow_html=True)
        else:
            # Mask the prediction where the value is 0 (background)
            combined_img = np.ma.masked_where(pred_slice == 0, pred_slice)

            # Custom colormap for the prediction mask
            custom_colorscale = [
                [0.0, 'rgba(0,0,0,0)'],  # Fully transparent where there is no tumor (background)
                [0.33, 'green'],  # Start with green (skipping blue)
                [0.66, 'yellow'],  # Then yellow
                [1.0, 'red']       # End with red
            ]

            # Custom colormap for the background slice
            custom_colorscale2 = [
                [0.0, 'black'],   # Black where there is no tumor (background)
                [0.33, 'green'],  # Start with green
                [0.66, 'yellow'], # Then yellow
                [1.0, 'red']      # End with red
            ]

            # Create subplots with 1 row and 2 columns
            figs = sp.make_subplots(rows=1, cols=2, subplot_titles=("Prediction", "T1c + Prediction Overlay"))

            # Add the prediction heatmap to the first subplot using the custom colorscale
            figs.add_trace(go.Heatmap(z=pred_slice, colorscale=custom_colorscale2, showscale=False), row=1, col=1)

            # Add grayscale MRI slice to the second subplot
            figs.add_trace(go.Heatmap(
                z=t1c_slice, 
                colorscale='gray', 
                showscale=False,
                hoverinfo='skip'  # Disable hover info for the background image
            ), row=1, col=2)

            # Add overlay with the prediction mask using the custom colormap to the second subplot
            figs.add_trace(go.Heatmap(
                z=combined_img, 
                colorscale=custom_colorscale, 
                showscale=False, 
                opacity=0.6,  # Set transparency
                hoverinfo='skip'  # Disable hover info for the overlay
            ), row=1, col=2)

            # Update layout
            figs.update_layout(
                height=400, 
                width=700, 
                title_text="Prediction and T1c + Prediction Overlay",
                title_font_color="white",
                paper_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
                plot_bgcolor='rgba(0, 0, 0, 0)',   # Transparent plot area
                font=dict(color="white")           # White font for contrast
            )

            # Display the combined plot
            st.plotly_chart(figs)

        if uploaded_files and file_names == st.session_state['prev_uploaded_files']:
            if gt_file:
                imageww = nib.load(gt_file.name).get_fdata()
                mask_slicec = imageww[56:184, 56:184, 13:141]
                # Extract the ground truth slice
                mask_slice = mask_slicec[:, n_slice, :].T  # Adjust this based on ground truth array dimensions
                # mask_slice = mask_gt[0, :, :, n_slice]  # Adjust this based on ground truth array dimensions
                # st.write('You selected:', np.unique(mask_gt))

                if np.all(mask_slice == 0):
                    st.markdown('<p style="color:green; font-weight:bold;">The ground truth indicates the absence of a tumor.</p>', unsafe_allow_html=True)
                else:

                    custom_colorscale2 = [
                        [0.0, 'black'],   # Black where there is no tumor (background)
                        [0.33, 'green'],  # Start with green
                        [0.66, 'yellow'], # Then yellow
                        [1.0, 'red']      # End with red
                    ]

                    custom_colorscale_gt = [
                        [0.0, 'rgba(0,0,0,0)'],  # Fully transparent where there is no tumor (background)
                        [0.33, 'green'],  # Start with green
                        [0.66, 'yellow'],  # Then yellow
                        [1.0, 'red']       # End with red
                    ]
                    # Create a new row for ground truth comparison
                    figs_gt = sp.make_subplots(rows=1, cols=2, subplot_titles=("Ground Truth", "T1c + Ground Truth Overlay"))

                    # Add the ground truth mask to the first subplot using the custom colormap
                    figs_gt.add_trace(go.Heatmap(z=mask_slice, colorscale=custom_colorscale2, showscale=False), row=1, col=1)

                    # Add overlay with the ground truth mask using the custom colormap to the second subplot
                    figs_gt.add_trace(go.Heatmap(
                        z=t1c_slice,
                        colorscale='gray',
                        showscale=False,
                        hoverinfo='skip'
                    ), row=1, col=2)

                    figs_gt.add_trace(go.Heatmap(
                        z=np.ma.masked_where(mask_slice == 0, mask_slice),  # Mask the ground truth
                        colorscale=custom_colorscale_gt,
                        showscale=False,
                        opacity=0.6,  # Set transparency
                        hoverinfo='skip'
                    ), row=1, col=2)

                    # Update layout
                    figs_gt.update_layout(
                        height=400,
                        width=700,
                        title_text="Ground Truth and T1c + Ground Truth Overlay",
                        title_font_color="white",
                        paper_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
                        plot_bgcolor='rgba(0, 0, 0, 0)',   # Transparent plot area
                        font=dict(color="white")           # White font for contrast
                    )

                    # Display the ground truth plot
                    st.plotly_chart(figs_gt)


    with tab6 : 

        n_slice = st.slider("Choose number of Slice to display", min_value=1, max_value=127, value=80, step=1, key="slider3")

        # Extract the slices for plotting
        pred_slice = val_pred[0, n_slice, :, :].T
        imagew = np.load(img_path)
        t1c_slice = imagew[1, :, :, :] 
        t1c_slice = t1c_slice[n_slice, :, :].T


        if np.all(pred_slice == 0):
            st.markdown('<p style="color:green; font-weight:bold;">No tumor detected in this slice.</p>', unsafe_allow_html=True)
        else:
            # Mask the prediction where the value is 0 (background)
            combined_img = np.ma.masked_where(pred_slice == 0, pred_slice)

            # Custom colormap for the prediction mask
            custom_colorscale = [
                [0.0, 'rgba(0,0,0,0)'],  # Fully transparent where there is no tumor (background)
                [0.33, 'green'],  # Start with green (skipping blue)
                [0.66, 'yellow'],  # Then yellow
                [1.0, 'red']       # End with red
            ]

            # Custom colormap for the background slice
            custom_colorscale2 = [
                [0.0, 'black'],   # Black where there is no tumor (background)
                [0.33, 'green'],  # Start with green (skipping blue)
                [0.66, 'yellow'], # Then yellow
                [1.0, 'red']      # End with red
            ]

            # Create subplots with 1 row and 2 columns
            figs = sp.make_subplots(rows=1, cols=2, subplot_titles=("Prediction", "T1c + Prediction Overlay"))

            # Add the prediction heatmap to the first subplot using the custom colorscale
            figs.add_trace(go.Heatmap(z=pred_slice, colorscale=custom_colorscale2, showscale=False), row=1, col=1)

            # Add grayscale MRI slice to the second subplot
            figs.add_trace(go.Heatmap(
                z=t1c_slice, 
                colorscale='gray', 
                showscale=False,
                hoverinfo='skip'  # Disable hover info for the background image
            ), row=1, col=2)

            # Add overlay with the prediction mask using the custom colormap to the second subplot
            figs.add_trace(go.Heatmap(
                z=combined_img, 
                colorscale=custom_colorscale, 
                showscale=False, 
                opacity=0.6,  # Set transparency
                hoverinfo='skip'  # Disable hover info for the overlay
            ), row=1, col=2)

            # Update layout
            figs.update_layout(
                height=400, 
                width=700, 
                title_text="Prediction and T1c + Prediction Overlay",
                title_font_color="white",
                paper_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
                plot_bgcolor='rgba(0, 0, 0, 0)',   # Transparent plot area
                font=dict(color="white")           # White font for contrast
            )

            # Display the combined plot
            st.plotly_chart(figs)

        if uploaded_files and file_names == st.session_state['prev_uploaded_files']:
            if gt_file:
                imageww = nib.load(gt_file.name).get_fdata()
                mask_slicec = imageww[56:184, 56:184, 13:141]
                # Extract the ground truth slice
                mask_slice = mask_slicec[n_slice, : , :].T  # Adjust this based on ground truth array dimensions
                # mask_slice = mask_gt[0, :, :, n_slice]  # Adjust this based on ground truth array dimensions
                # st.write('You selected:', np.unique(mask_gt))

                if np.all(mask_slice == 0):
                    st.markdown('<p style="color:green; font-weight:bold;">The ground truth indicates the absence of a tumor.</p>', unsafe_allow_html=True)
                else:

                    custom_colorscale2 = [
                        [0.0, 'black'],   # Black where there is no tumor (background)
                        [0.33, 'green'],  # Start with green
                        [0.66, 'yellow'], # Then yellow
                        [1.0, 'red']      # End with red
                    ]

                    custom_colorscale_gt = [
                        [0.0, 'rgba(0,0,0,0)'],  # Fully transparent where there is no tumor (background)
                        [0.33, 'green'],  # Start with green
                        [0.66, 'yellow'],  # Then yellow
                        [1.0, 'red']       # End with red
                    ]
                    # Create a new row for ground truth comparison
                    figs_gt = sp.make_subplots(rows=1, cols=2, subplot_titles=("Ground Truth", "T1c + Ground Truth Overlay"))

                    # Add the ground truth mask to the first subplot using the custom colormap
                    figs_gt.add_trace(go.Heatmap(z=mask_slice, colorscale=custom_colorscale2, showscale=False), row=1, col=1)

                    # Add overlay with the ground truth mask using the custom colormap to the second subplot
                    figs_gt.add_trace(go.Heatmap(
                        z=t1c_slice,
                        colorscale='gray',
                        showscale=False,
                        hoverinfo='skip'
                    ), row=1, col=2)

                    figs_gt.add_trace(go.Heatmap(
                        z=np.ma.masked_where(mask_slice == 0, mask_slice),  # Mask the ground truth
                        colorscale=custom_colorscale_gt,
                        showscale=False,
                        opacity=0.6,  # Set transparency
                        hoverinfo='skip'
                    ), row=1, col=2)

                    # Update layout
                    figs_gt.update_layout(
                        height=400,
                        width=700,
                        title_text="Ground Truth and T1c + Ground Truth Overlay",
                        title_font_color="white",
                        paper_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
                        plot_bgcolor='rgba(0, 0, 0, 0)',   # Transparent plot area
                        font=dict(color="white")           # White font for contrast
                    )

                    # Display the ground truth plot
                    st.plotly_chart(figs_gt)

        st.session_state['prev_uploaded_files'] = file_names


    
else:
    st.write("Make sure you image is in nii/nii.gz Format.")

