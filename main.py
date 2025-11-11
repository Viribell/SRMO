import os
import numpy as np

os.environ[ 'TF_CPP_MIN_LOG_LEVEL' ] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

g_Model = None


#-----------------------------------------MAIN_LOWER_FUNC

#-----------------------------------------MAIN_UPPER_FUNC
def InitSystem():
    global g_Model


    print( "\n\n\n" )


#-----------------------------------------MAIN

InitSystem()