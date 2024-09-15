import numpy as np
from torch.utils.data import DataLoader


def imageLoader(img_path, batch_size, num_workers=24):
    
    dataset = np.load(img_path)

    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers, 
                            pin_memory=True)
                            
    return dataloader
