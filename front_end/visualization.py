import nibabel as nib
import numpy as np
import imageio
from PIL import Image
import os

def resize_image(image, width, height):
    return image.resize((width, height), Image.BILINEAR)

def create_rotated_coronal_gif_from_nifti(nifti_file, gif_file, fps=10):
    # Load the NIfTI file
    nifti_img = nib.load(nifti_file)
    img_data = nifti_img.get_fdata()
    
    # Determine the dimensions
    x_dim, y_dim, z_dim = img_data.shape
    
    # Create a list to hold the file paths of the images
    images = []

    for i in range(y_dim):
        # Extract each coronal slice
        coronal_slice = img_data[:, i, :]
        
        # Normalize the slice to the range [0, 255] and convert to uint8
        norm_slice = np.uint8(255 * (coronal_slice - np.min(coronal_slice)) / (np.max(coronal_slice) - np.min(coronal_slice)))
        
        # Convert the numpy array to a PIL image
        pil_img = Image.fromarray(norm_slice)
        
        # Rotate the image 90 degrees counterclockwise
        rotated_img = pil_img.rotate(90, expand=True)

        resized_img = resize_image(rotated_img, 580, 380)

        
        # Create a temporary file to save the image
        temp_filename = f'temp_coronal_{i}.png'
        resized_img.save(temp_filename)
        images.append(temp_filename)
    
    # Read the images and create a GIF
    with imageio.get_writer(gif_file, mode='I', duration=1/fps,loop=0) as writer:
        for image in images:
            img = imageio.imread(image)
            writer.append_data(img)
    
    # Clean up temporary files
    for image in images:
        os.remove(image)



def create_rotated_axial_gif_from_nifti(nifti_file, gif_file, fps=10):
    # Load the NIfTI file
    nifti_img = nib.load(nifti_file)
    img_data = nifti_img.get_fdata()
    
    # Determine the dimensions
    x_dim, y_dim, z_dim = img_data.shape
    
    # Create a list to hold the file paths of the images
    images = []

    for i in range(z_dim):
        # Extract each axial slice
        axial_slice = img_data[:, :, i]
        
        # Normalize the slice to the range [0, 255] and convert to uint8
        norm_slice = np.uint8(255 * (axial_slice - np.min(axial_slice)) / (np.max(axial_slice) - np.min(axial_slice)))
        
        # Convert the numpy array to a PIL image
        pil_img = Image.fromarray(norm_slice)
        
        # Rotate the image 90 degrees counterclockwise
        # rotated_img = pil_img.rotate(90, expand=True)
        resized_img = resize_image(pil_img, 580, 380)

        
        # Create a temporary file to save the image
        temp_filename = f'temp_axial_{i}.png'
        resized_img.save(temp_filename)
        images.append(temp_filename)
    
    # Read the images and create a GIF
    with imageio.get_writer(gif_file, mode='I', duration=1/fps,loop=0) as writer:
        for image in images:
            img = imageio.imread(image)
            writer.append_data(img)
    
    # Clean up temporary files
    for image in images:
        os.remove(image)




def create_rotated_sagittal_gif_from_nifti(nifti_file, gif_file, fps=10):
    # Load the NIfTI file
    nifti_img = nib.load(nifti_file)
    img_data = nifti_img.get_fdata()
    
    # Determine the dimensions
    x_dim, y_dim, z_dim = img_data.shape
    
    # Create a list to hold the file paths of the images
    images = []

    for i in range(x_dim):
        # Extract each sagittal slice
        sagittal_slice = img_data[i, :, :]
        
        # Normalize the slice to the range [0, 255] and convert to uint8
        norm_slice = np.uint8(255 * (sagittal_slice - np.min(sagittal_slice)) / (np.max(sagittal_slice) - np.min(sagittal_slice)))
        
        # Convert the numpy array to a PIL image
        pil_img = Image.fromarray(norm_slice)
        
        # Rotate the image 90 degrees counterclockwise
        rotated_img = pil_img.rotate(90, expand=True)

        resized_img = resize_image(rotated_img, 580, 380)

        
        # Create a temporary file to save the image
        temp_filename = f'temp_sagittal_{i}.png'
        resized_img.save(temp_filename)
        images.append(temp_filename)
    
    # Read the images and create a GIF
    with imageio.get_writer(gif_file, mode='I', duration=1/fps,loop=0) as writer:
        for image in images:
            img = imageio.imread(image)
            writer.append_data(img)
    
    # Clean up temporary files
    for image in images:
        os.remove(image)