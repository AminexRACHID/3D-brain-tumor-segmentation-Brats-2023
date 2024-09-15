import glob
import nibabel as nib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn.functional as F
import random

# Check if CUDA (GPU) is available and use it, otherwise, fall back to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f"Using",device)

# Function to scale an image using MinMaxScaler
def scale_image(image):
    scaler = MinMaxScaler()
    image = scaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)
    return image

def generate_random_name():
    part1 = "BraTS"
    part2 = f"{random.randint(10000, 99999)}"  # 5-digit random number
    part3 = f"{random.randint(100, 999)}"      # 3-digit random number
    random_name = f"{part1}-{part2}-{part3}"
    return random_name

# Function to convert the .nii.gz files to .npy combined
# Operation - Scaling, Cropping, Combining, Permutation (for PyTorch Models)
def nnunet_to_npy(file_info_list):

    # T1n image
    temp_image_t1n = nib.load(file_info_list.get("t1n")).get_fdata()
    temp_image_t1n = scale_image(temp_image_t1n)

    # T1c image
    temp_image_t1c = nib.load(file_info_list.get("t1c")).get_fdata()
    temp_image_t1c = scale_image(temp_image_t1c)

    # T2w image
    temp_image_t2w = nib.load(file_info_list.get("t2w")).get_fdata()
    temp_image_t2w = scale_image(temp_image_t2w)

    # T2f image
    temp_image_t2f = nib.load(file_info_list.get("t2f")).get_fdata()
    temp_image_t2f = scale_image(temp_image_t2f)

    # Cropping the images and the mask
    temp_image_t1n = temp_image_t1n[56:184, 56:184, 13:141]
    temp_image_t1c = temp_image_t1c[56:184, 56:184, 13:141]
    temp_image_t2w = temp_image_t2w[56:184, 56:184, 13:141]
    temp_image_t2f = temp_image_t2f[56:184, 56:184, 13:141]

    # Convert NumPy arrays to PyTorch tensors and move them to the GPU if available
    temp_image_t1n = torch.tensor(temp_image_t1n, dtype=torch.float32, device=device)
    temp_image_t1c = torch.tensor(temp_image_t1c, dtype=torch.float32, device=device)
    temp_image_t2w = torch.tensor(temp_image_t2w, dtype=torch.float32, device=device)
    temp_image_t2f = torch.tensor(temp_image_t2f, dtype=torch.float32, device=device)

    # Stack the images into a single volume
    temp_combined_images = torch.stack([temp_image_t1n, temp_image_t1c, temp_image_t2w, temp_image_t2f], dim=0)
    
    random_name = generate_random_name()    
    # Save the images and the mask
    np.save(random_name+".npy", temp_combined_images.cpu().numpy())
                
    # Free up GPU memory after processing
    torch.cuda.empty_cache()
    return random_name


def mask_preprocessing(mask_name):

    # T1n image
    temp_mask = nib.load(mask_name).get_fdata()
    temp_mask = temp_mask.astype(np.uint8)
    temp_mask[temp_mask == 4] = 3
    temp_mask = temp_mask.astype(np.float64)

    temp_mask = temp_mask[56:184, 56:184, 13:141]

    temp_mask = torch.tensor(temp_mask, dtype=torch.long, device=device)

    temp_mask = F.one_hot(temp_mask, num_classes=4).permute(3, 0, 1, 2)

    random_name = generate_random_name()    

    np.save(random_name+".npy", temp_mask.cpu().numpy().astype(np.uint8))
                
    # Free up GPU memory after processing
    torch.cuda.empty_cache()
    return random_name

            




