import training
import cebra
import numpy as np
import torch
import cebra_utils
import os
import re
import tifffile
import cv2
import torchvision

## Given the path to a tif file, return that as a 3d numpy array
# @param path: path to tif file
# @return: 3d numpy array, first array is time dimension
def load_tif(path):
    img = tifffile.imread(path)
    img = np.array(img)
    return img

## Loads the brain data from a given trial
def load_brain_data(parent_directory, trial_num, type='gcamp'):
    # Load the data
    data_path = os.path.join(parent_directory, 'trial_' + str(trial_num) + '/brain/' + type + '.tif')
    data = load_tif(data_path)
    return data

## Creat and train the model in partial batches of data
def train_model(model, dataloader, criterion, loader_type, device, output_model_path):

    ## Load criterion
    criterion.to(device)
    ## Load optimizer
    optimizer = torch.optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=0.001)

    print('Loading solver')
    ## Load solver and train on first slice of data
    if loader_type == 'multisession':
        solver = cebra.solver.MultiSessionSolver(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            tqdm_on=True,
        ).to(device)

    elif loader_type == 'single':
        solver = cebra.solver.SingleSessionSolver(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            tqdm_on=True,
        ).to(device)

    solver.fit(dataloader.to(device),
                save_frequency=5000,
                logdir='runs',)
 
    print('Training complete, saving model')
    torch.save(solver, output_model_path)
    return solver

## From a given path load and then flatten the trial and brain data and do a test / train split
# @param data_path: path to the data
# @param split: tuple range of data to be used for training
# @return: flattened brain data, flattened feature data, discrete training data, testing brain data, testing feature data
def load_test_train(data_path, split, image_size=64, use_pose=False, embedding_path=None):

    num_trials = len([x for x in os.listdir(data_path) if re.match('trial_[0-9]+$', x)])
    split_indices = np.arange(num_trials * split[0], num_trials * split[1])
    split_indices = [int(x) for x in split_indices]
    brain_data = [load_brain_data(data_path, x) for x in split_indices]
    if use_pose == True:
        feature_data = [training.load_pose_data(data_path, x) for x in split_indices]
    else:
        feature_data = [training.load_embedding_data(embedding_path, x) for x in split_indices]
    # flatten the first dimension of brain data
    # n x 288 x 256 x 256 -> n * 288 x 256 x 256
    # before flattening take train test split

    flattened_brain_data = np.concatenate(brain_data, axis=0)
    flattened_feature_data = np.concatenate(feature_data, axis=0)
    # create a discrete tensor for the brain data of 0-288 repeating
    discrete_training_data = np.concatenate(np.array([np.arange(288) for _ in range(len(brain_data))]), axis=0)
    return flattened_brain_data, flattened_feature_data, discrete_training_data

## Load Train data on multiple trials and concatenate them
# @param data_paths: list of paths to the data
# @param split: the percentage of data to be used for training
# @return: flattened brain data, flattened feature data, discrete training data, testing brain data, testing feature data
def load_train_data_multi(data_paths, split, image_size=64, use_pose=False, embedding_paths=None):
    for i, data_path in enumerate(data_paths):
        print('Loading data from ' + data_path)
        flattened_brain_data, flattened_feature_data, discrete_training_data = load_test_train(data_path, split, image_size, use_pose, embedding_paths[i])
        if i == 0:
            brain_data = flattened_brain_data
            feature_data = flattened_feature_data
            discrete_data = discrete_training_data
        else:
            brain_data = np.concatenate((brain_data, flattened_brain_data), axis=0)
            feature_data = np.concatenate((feature_data, flattened_feature_data), axis=0)
            discrete_data = np.concatenate((discrete_data, discrete_training_data), axis=0)
    return brain_data, feature_data, discrete_data, None, None

## Main training loop
if __name__ == "__main__":



    ## Multi Session Data Loading
    data_paths = [
        '/mnt/teams/TM_Lab/Tony/water_reaching/Data/rig1_data/processed/FRM1_2023-07-07_1',
        '/mnt/teams/TM_Lab/Tony/water_reaching/Data/rig1_data/processed/FRM1_2023-06-24_1',
        '/mnt/teams/TM_Lab/Tony/water_reaching/Data/rig1_data/processed/FRM1_2023-06-25_1',
        '/mnt/teams/TM_Lab/Tony/water_reaching/Data/rig1_data/processed/FRM1_2023-06-27_1',
    ]

    embedding_paths = [
        '/home/murph_4090ws/Documents/Water_Reaching_Classifier/FRM1_7-07',
        '/home/murph_4090ws/Documents/Water_Reaching_Classifier/FRM1_6-24',
        '/home/murph_4090ws/Documents/Water_Reaching_Classifier/FRM1_6-25',
        '/home/murph_4090ws/Documents/Water_Reaching_Classifier/FRM1_6-27',
    ]

    ## For Training Masked Model
    Model_2D = True
    Model_Name = "masked-transformer"
    Image_Size = None
    depth = 128
    output_model_path='masked-transformer.pth'
    mask_path = '/mnt/teams/TM_Lab/Tony/water_reaching/Data/rig1_data/processed/FRM1_2023-07-07_1/mask.csv'
    model = cebra.models.init(
        Model_Name, 
        num_neurons=10,
        num_units=depth,
        num_output=8,
        masks=cebra_utils.load_masks(mask_path).to('cuda')
    ).to('cuda')

    criteria = cebra.models.criterions.LearnableCosineInfoNCE(temperature=1, min_temperature=0.2)
    ## Batched training data loading
    # create list of splits to train on
    n_splits = 20
    split_centers = np.linspace(0, 1, n_splits + 1)[1:-1]
    split_width = 0.1
    splits = [(center - split_width / 2, center + split_width / 2) for center in split_centers]
    ## clip between 0 and 1
    splits = [(max(0, start), min(1, end)) for start, end in splits]
    ## tile splits 10 times to get 200 splits
    splits = np.tile(splits, (10, 1))
    ## randomize the splits
    np.random.shuffle(splits)
    flattened_brain_data_all, flattened_feature_data_all, discrete_data__all, _, _ = load_train_data_multi(data_paths, (0,1), Image_Size, use_pose=False, embedding_paths=embedding_paths)
    for split in splits:

        ## partition the data based off current split
        flattened_brain_data = flattened_brain_data_all[int(len(flattened_brain_data_all) * split[0]):int(len(flattened_brain_data_all) * split[1])]
        flattened_feature_data = flattened_feature_data_all[int(len(flattened_feature_data_all) * split[0]):int(len(flattened_feature_data_all) * split[1])]
        discrete_data = discrete_data__all[int(len(discrete_data__all) * split[0]):int(len(discrete_data__all) * split[1])]

        loader = training.init_single_session_dataloader(
            brain_data=flattened_brain_data,
            feature_data=flattened_feature_data,
            discrete_data=discrete_data,
            num_steps=500,
            time_offset=30,
            conditional='time_delta',
            batch_size=128,
            cebra_offset=cebra.data.datatypes.Offset(0,1),
        )
        print('Training on slices ' + str(split))
        # For ViT model we need to reshape the data to be 256 x 256 x 3 as the model expects 3 channels, so we use a 1,2 offset
        train_model(
            model=model, 
            dataloader=loader,
            criterion=criteria, 
            loader_type='single',
            device='cuda',
            output_model_path=output_model_path
        )