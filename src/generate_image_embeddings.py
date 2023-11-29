import sys
import numpy as np
import matplotlib.pyplot as plt
import cebra
from PIL import Image
import cv2
import os
import torch
import itertools
import random
import gc
import pandas as pd
import argparse
import torchvision.models as models

## Use VGG 16 to generate image embeddings for given videos and save them to disk at specified path ##
## @param path: path to the directory containing the videos
## @param trial: trial number
## @return: 3 numpy arrays, each containing the frames from a different camera, A, B, and C
def load_mp4(path, trial):
    video_path_A = path + '/trial_' + str(trial) + '/camA/trimmed.mp4'
    video_path_B = path + '/trial_' + str(trial) + '/camB/trimmed.mp4'
    video_path_C = path + '/trial_' + str(trial) + '/camC/trimmed.mp4'

    capA = cv2.VideoCapture(video_path_A)
    capB = cv2.VideoCapture(video_path_B)
    capC = cv2.VideoCapture(video_path_C)  

    ## Iterate through each video and save the frames to a numpy array
    framesA = []
    framesB = []
    framesC = []

    while(capA.isOpened()):
        ret, frame = capA.read()
        if ret == False:
            break
        framesA.append(frame)
    capA.release()

    while(capB.isOpened()):
        ret, frame = capB.read()
        if ret == False:
            break
        framesB.append(frame)
    capB.release()

    while(capC.isOpened()):
        ret, frame = capC.read()
        if ret == False:
            break
        framesC.append(frame)
    capC.release()

    ## Convert the frames to a numpy array
    framesA = np.array(framesA)
    framesB = np.array(framesB)
    framesC = np.array(framesC)

    # return frame
    return framesA, framesB, framesC

## Takes a model and a numpy array of frames and returns a numpy array of image embeddings
## @param model: model to use to generate the embeddings
## @param frames: numpy array of frames
## @param size: size to resize the frames to before generating the embeddings
## @return: numpy array of image embeddings
def generate_image_embeddings(model, frames, size=(224,224)):
    # Resize the frames
    frames = np.array([cv2.resize(frame, size) for frame in frames])
    # Convert the frames to a tensor
    frames = torch.FloatTensor(frames).to('cuda')
    # Reshape the frames to be in the correct format
    frames = frames.permute(0,3,1,2)
    # Generate the embeddings
    embeddings = model(frames)
    # Convert the embeddings to a numpy array
    embeddings = embeddings.cpu().detach().numpy()
    return embeddings

parser = argparse.ArgumentParser(description='Generate image embeddings for given videos and save them to disk at specified path')
parser.add_argument('--path', type=str, help='path to the directory containing the videos')
parser.add_argument('--dest', type=str, help='path to save the image embeddings')

if __name__ == '__main__':
    args = parser.parse_args()
    path = args.path
    dest = args.dest

    # if destination path does not exist, create it
    if not os.path.exists(dest):
        os.makedirs(dest)

    ## Get the number of trials
    num_trials = len([x for x in os.listdir(path) if 'trial_' in x])

    ## Load the model (in this case VGG 16)
    vgg16 = models.vgg16(pretrained=True)
    ## Remove the last layer
    vgg16.classifier = torch.nn.Sequential(*list(vgg16.classifier.children())[:-1])
    ## Set the model to eval mode
    vgg16.eval().to('cuda')

    ## Load the data
    for trial_num in range(num_trials):
        print('Loading trial ' + str(trial_num))
        framesA, framesB, framesC = load_mp4(path, trial_num)
        print('Generating embeddings for trial ' + str(trial_num))
        embeddingsA = generate_image_embeddings(vgg16, framesA)
        embeddingsB = generate_image_embeddings(vgg16, framesB)
        embeddingsC = generate_image_embeddings(vgg16, framesC)
        print('Saving embeddings for trial ' + str(trial_num))
        ## Concate all 3 embeddings into a single numpy array
        embeddings_all = np.concatenate((embeddingsA, embeddingsB, embeddingsC), axis=1)
        print(np.shape(embeddings_all))
        np.save(dest + '/trial_' + str(trial_num) + '_embeddings.npy', embeddings_all)
