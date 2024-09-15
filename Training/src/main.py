import os, sys
import pandas as pd
import torch
import torch.optim as optim
import torch.cuda.amp as amp
from data_generator import imageLoader
from models import SAD_UNet
from losses import dice_loss, accuracy, sensitivity, specificity, dice_score, precision
from losses import CombinedLoss
from multiprocessing import freeze_support

def main():

    TRAIN_IMG_DIR = "./data/training/images"
    TRAIN_MASK_DIR = "./data/training/masks"

    VAL_IMG_DIR = "./data/test/images"
    VAL_MASK_DIR = "./data/test/masks"


    # Get list of image and mask names
    train_img_list = sorted(os.listdir(TRAIN_IMG_DIR))
    train_mask_list = sorted(os.listdir(TRAIN_MASK_DIR))

    # Remove files not ending with ".npy"
    train_img_list = [file for file in train_img_list if file.endswith('.npy')]
    train_mask_list = [file for file in train_mask_list if file.endswith('.npy')]

    # Check Count
    if(len(train_img_list) != len(train_mask_list)):
        print("Images and Masks not equal in size")
        input()

    # Get list of image and mask names
    val_img_list = sorted(os.listdir(VAL_IMG_DIR))
    val_mask_list = sorted(os.listdir(VAL_MASK_DIR))

    val_img_list = [file for file in val_img_list if file.endswith('.npy')]
    val_mask_list = [file for file in val_mask_list if file.endswith('.npy')]

    # Check Count
    if(len(val_img_list) != len(val_mask_list)):
        print("Validation Images and Masks not equal in size")
        input()


    # Print the sizes of the training and validation sets
    print(f"Number of training images: {len(train_img_list)}")
    print(f"Number of validation images: {len(val_img_list)}")

   
    class_weights = {0: 0.26, 1: 33.18, 2: 8.57, 3: 23.45}

    # Set Learning rate
    LR = 0.00005

    # Define Batch Size
    BATCH_SIZE = 1 
    accumulation_steps = 5 

    # Specify GPU ids to use
    gpus = [0] 
    device = torch.device('cuda:{}'.format(gpus[0]) if torch.cuda.is_available() else 'cpu')

    # Check if CUDA is available
    print('Using device:', device)

    torch.cuda.empty_cache()

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Define the model and move it to the GPU
    model = SAD_UNet(in_channels=4, num_classes=4)

    # Move model to GPU
    model = model.to(device)

    print("model created")

    num_params = count_parameters(model)
    print(f"Number of parameters in the model: {num_params}")

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR)


    class_weights_list = [class_weights[i] for i in range(len(class_weights))]
    criterion = CombinedLoss(num_classes=3, class_weights=class_weights_list, device_num = gpus[0]).to(device)

    # Define the DataLoaders
    train_data_gen = imageLoader(TRAIN_IMG_DIR, train_img_list, TRAIN_MASK_DIR, train_mask_list, BATCH_SIZE, num_workers = 8)
    val_data_gen = imageLoader(VAL_IMG_DIR, val_img_list, VAL_MASK_DIR, val_mask_list, BATCH_SIZE, num_workers=8)


    # Initialize the gradient scaler for amp
    scaler = amp.GradScaler()


    print("Model Training Step")

    # Initialize results DataFrame
    results_df = pd.DataFrame(columns=[
        'Epoch', 'Loss', 'Accuracy',
        'Dice Coef (0)', 'Dice Coef (1)', 'Dice Coef (2)', 'Dice Coef (3)',
        'Sensitivity (0)', 'Sensitivity (1)', 'Sensitivity (2)', 'Sensitivity (3)',
        'Specificity (0)', 'Specificity (1)', 'Specificity (2)', 'Specificity (3)',
        'Precision (0)', 'Precision (1)', 'Precision (2)', 'Precision (3)'
    ])
    freeze_support()
    # Training loop
    best_loss = float('inf')
    num_classes = 4
    for epoch in range(250):
        model.train()
        epoch_loss = 0


        # Metrics initialization
        total_accuracy = 0
        total_sensitivity = [0] * num_classes
        total_specificity = [0] * num_classes
        total_precision = [0] * num_classes
        total_dice_score = [0] * num_classes


        for batch_num, (imgs, masks) in enumerate(train_data_gen):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()

            with amp.autocast():
                main_logit, aux_logit_1, aux_logit_2, aux_logit_3, aux_logit_4 = model(imgs)
                masks = torch.argmax(masks, dim=1)  # To remove one-hot encoding

                # Compute loss for each of the outputs
                main_loss = criterion(main_logit, masks)
                aux_loss_1 = criterion(aux_logit_1, masks)
                aux_loss_2 = criterion(aux_logit_2, masks)
                aux_loss_3 = criterion(aux_logit_3, masks)
                aux_loss_4 = criterion(aux_logit_4, masks)

                # Average the losses
                loss = main_loss + aux_loss_1 + aux_loss_2 + aux_loss_3 + aux_loss_4
                loss /= 5

                outputs = torch.argmax(main_logit, dim=1) # To remove one-hot encoding

                # Compute the mean loss for optimization
                loss = loss.mean()

            # Backward pass and optimization
            scaler.scale(loss).backward(retain_graph=True)

            # Perform optimization after accumulation_steps
            if (batch_num + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            print("Batch:", batch_num,", Combined Loss:", loss,", Dice Loss:", dice_loss(masks, outputs).mean().item())

            epoch_loss += loss.item()

            # Calculate metrics for each class
            total_accuracy += accuracy(outputs, masks).item()
            for class_id in range(num_classes):
                total_sensitivity[class_id] += sensitivity(outputs, masks, class_id).item()
                total_specificity[class_id] += specificity(outputs, masks, class_id).item()
                total_precision[class_id] += precision(outputs, masks, class_id).item()
                total_dice_score[class_id] += dice_score(outputs, masks, class_id).item()


        # Divide by the number of classes to get the average
        epoch_loss /= len(train_data_gen)
        total_accuracy /= len(train_data_gen)
        total_sensitivity = [sens / len(train_data_gen) for sens in total_sensitivity]
        total_specificity = [spec / len(train_data_gen) for spec in total_specificity]
        total_precision = [prec / len(train_data_gen) for prec in total_precision]
        total_dice_score = [dice / len(train_data_gen) for dice in total_dice_score]

        print(f"----------Epoch {epoch+1}, Loss: {epoch_loss}, Accuracy: {total_accuracy}, Dice Coef: {total_dice_score}, "
            f"Sensitivity: {total_sensitivity}, "
            f"Specificity: {total_specificity}, "
            f"Precision: {total_precision}")

        # Append results to DataFrame
        new_row = {'Epoch': epoch + 1,
                'Loss': epoch_loss,
                'Accuracy': total_accuracy,
                'Dice Coef (0)': total_dice_score[0],
                'Dice Coef (1)': total_dice_score[1],
                'Dice Coef (2)': total_dice_score[2],
                'Dice Coef (3)': total_dice_score[3],
                'Sensitivity (0)': total_sensitivity[0],
                'Sensitivity (1)': total_sensitivity[1],
                'Sensitivity (2)': total_sensitivity[2],
                'Sensitivity (3)': total_sensitivity[3],
                'Specificity (0)': total_specificity[0],
                'Specificity (1)': total_specificity[1],
                'Specificity (2)': total_specificity[2],
                'Specificity (3)': total_specificity[3],
                'Precision (0)': total_precision[0],
                'Precision (1)': total_precision[1],
                'Precision (2)': total_precision[2],
                'Precision (3)': total_precision[3]}

        results_df = pd.concat([results_df, pd.DataFrame(new_row, index=[0])], ignore_index=True)

        # Validation step after every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval() 

            # Initialize metrics for validation
            val_loss = 0
            val_total_accuracy = 0
            val_total_sensitivity = [0] * num_classes
            val_total_specificity = [0] * num_classes
            val_total_precision = [0] * num_classes
            val_total_dice_score = [0] * num_classes


            with torch.no_grad():  # Disable gradients for validation
                for val_batch_num, (val_imgs, val_masks) in enumerate(val_data_gen):
                    val_imgs, val_masks = val_imgs.to(device), val_masks.to(device)

                    val_main_logit, val_aux_logit_1, val_aux_logit_2, val_aux_logit_3, val_aux_logit_4 = model(val_imgs) 
                    val_masks = torch.argmax(val_masks, dim=1) 

                    # Compute loss for each of the outputs
                    val_main_loss = criterion(val_main_logit, val_masks)
                    val_aux_loss_1 = criterion(val_aux_logit_1, val_masks)
                    val_aux_loss_2 = criterion(val_aux_logit_2, val_masks)
                    val_aux_loss_3 = criterion(val_aux_logit_3, val_masks)
                    val_aux_loss_4 = criterion(val_aux_logit_4, val_masks)

                    # Average the losses
                    val_loss_batch = val_main_loss + val_aux_loss_1 + val_aux_loss_2 + val_aux_loss_3 + val_aux_loss_4
                    val_loss_batch /= 5

                    val_outputs = torch.argmax(val_main_logit, dim=1) 

                    # Compute the mean loss for optimization
                    val_loss += val_loss_batch.mean().item()

                    # Calculate metrics for validation
                    val_total_accuracy += accuracy(val_outputs, val_masks).item()
                    for class_id in range(num_classes):
                        val_total_sensitivity[class_id] += sensitivity(val_outputs, val_masks, class_id).item()
                        val_total_specificity[class_id] += specificity(val_outputs, val_masks, class_id).item()
                        val_total_precision[class_id] += precision(val_outputs, val_masks, class_id).item()
                        val_total_dice_score[class_id] += dice_score(val_outputs, val_masks, class_id).item()


            # Calculate average validation metrics
            val_loss /= len(val_data_gen)
            val_total_accuracy /= len(val_data_gen)
            val_total_sensitivity = [sens / len(val_data_gen) for sens in val_total_sensitivity]
            val_total_specificity = [spec / len(val_data_gen) for spec in val_total_specificity]
            val_total_precision = [sens / len(val_data_gen) for sens in val_total_precision]
            val_total_dice_score = [spec / len(val_data_gen) for spec in val_total_dice_score]


            # Print validation results
            print(f"-----------Validation Epoch {epoch+1}, Loss: {val_loss}, Accuracy: {val_total_accuracy}, Dice Coef: {val_total_dice_score}, "
                f"Sensitivity: {val_total_sensitivity}, "
                f"Specificity: {val_total_specificity}, "
                f"Precision: {val_total_precision}")
                # ... other metrics print statements ...

            # Append results to DataFrame
            new_row = {'Epoch': "Validation "+ str(epoch + 1),
                    'Loss': val_loss,
                    'Accuracy': val_total_accuracy,
                    'Dice Coef (0)': val_total_dice_score[0],
                    'Dice Coef (1)': val_total_dice_score[1],
                    'Dice Coef (2)': val_total_dice_score[2],
                    'Dice Coef (3)': val_total_dice_score[3],
                    'Sensitivity (0)': val_total_sensitivity[0],
                    'Sensitivity (1)': val_total_sensitivity[1],
                    'Sensitivity (2)': val_total_sensitivity[2],
                    'Sensitivity (3)': val_total_sensitivity[3],
                    'Specificity (0)': val_total_specificity[0],
                    'Specificity (1)': val_total_specificity[1],
                    'Specificity (2)': val_total_specificity[2],
                    'Specificity (3)': val_total_specificity[3],
                    'Precision (0)': val_total_precision[0],
                    'Precision (1)': val_total_precision[1],
                    'Precision (2)': val_total_precision[2],
                    'Precision (3)': val_total_precision[3]}

            results_df = pd.concat([results_df, pd.DataFrame(new_row, index=[0])], ignore_index=True)

            model.train()  # Switch back to training mode

        # Save the DataFrame to a CSV file after every epoch
        results_df.to_csv('training_results.csv', index=False)

        # Save the best model with lowest loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            print("Saving model")
            torch.save(model.state_dict(), './saved_model/SAD_UNet_model.pt')

    # Save the final model
    print("Saving final model")
    torch.save(model.state_dict(), './saved_model/final_SAD_UNet_model.pt')


if __name__ == '__main__':
    freeze_support()
    main()