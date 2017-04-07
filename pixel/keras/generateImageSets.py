"""Generate Tensors for Keras model learning
Example:
python generateImageSets --dataset=data/
"""

from PIL import Image as pil_image
import numpy as np
import argparse
import os
import string

def parse_args():
	"""
	Handles input of folder with all gifs with different resolution images
	"""
	parser = argparse.ArgumentParser(description="dataset folder")
	parser.add_argument('--dataset', dest="dataset", help="dataset path")
	return parser.parse_args()

def generate_GT_HR_sets(path):
    """
    Generates sets of ground-truth and high-res images for each frame of a gif
    Specifically each set is a 3-element list of 3-D matrices representing the
        GT and HR images.
    Each set is structured as [GT, HR, GT2] with GT being at frame n, and HR at
        frame n+1, and GT2 at frame n+1

    Input:
    path :  file path to directory with folders of gifs each with subfolders for
            lr/hr/gt images in numerical order of frames. 00000.jpg is the first
            frame

    Returns:
    5-D Tensor with a list of sets of 3D matrices
    """

    dataset = []
    for gifFolder in os.listdir(path):

        subdirs = os.listdir(path+"/"+gifFolder)
        if not('gt' in subdirs and 'lr' in subdirs and 'hr' in subdirs):
            raise ValueError('could not find gt lr and hr subdirs in %s'
                                % (path+"/"+gifFolder))
        hrImages = os.listdir(path+"/"+gifFolder+"/"+"hr")
        lrImages = os.listdir(path+"/"+gifFolder+"/"+"lr")
        gtImages = os.listdir(path+"/"+gifFolder+"/"+"gt")

        #Take up to 2nd to last image
        for pos in xrange(len(gtImages)-1):
            gtImage  = gtImages[pos]
            hrImage  = hrImages[pos+1]
            gtImage2 = gtImages[pos+1]

            imageSet = [gtImage, hrImage, gtImage2]


            # gtImage  = pil_image.open(path+"/"+gifFolder+"/"+"gt"+"/"+gtImage)
            # hrImage  = pil_image.open(path+"/"+gifFolder+"/"+"gt"+"/"+hrImage)
            # gtImage2 = pil_image.open(path+"/"+gifFolder+"/"+"gt"+"/"+gtImage2)

            for i,img in enumerate(imageSet):
                imageSet[i]=pil_image.open(path+"/"+gifFolder+"/"+"gt"+"/"+imageSet[i])
                imageSet[i]=imageSet[i].convert('RGB')
                imageSet[i]=np.asarray(imageSet[i], dtype=np.float32)

            # #If one image not RGB, assume all aren't
            # if gtImage.mode != 'RGB':
            #     gtImage  = gtImage.convert('RGB')
            #     hrImage  = hrImage.convert('RGB')
            #     gtImage2 = gtImage2.convert('RGB')
            #
            # gtImage  = np.asarray(gtImage, dtype=np.float32)
            # hrImage  = np.asarray(hrImage, dtype=np.float32)
            # gtImage2 = np.asarray(gtImage2, dtype=np.float32)

            dataset.append(imageSet)
    return np.array(dataset)

if __name__=='__main__':
	args = parse_args()
	dataFolder = args.dataset
	generate_GT_HR_sets(dataFolder)
