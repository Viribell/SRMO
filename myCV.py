import numpy as np
import cv2
from PIL import Image, ImageTk

g_cvFaceClassifierName = "haarcascade_frontalface_default.xml"

def cvGetCascadeClassifier( name ):
    l_Classifier = cv2.CascadeClassifier( cv2.data.haarcascades + name )

    return l_Classifier

def cvLoadImage( imgPath ):
    l_Img = cv2.imread( imgPath ) #in BGR scale by default

    return l_Img

def cvConvertImageToGrayscale( img ):
    l_GrayImg = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )

    return l_GrayImg

def cvResizeImage( img, imgSize ):
    l_Img = cv2.resize( img, imgSize )

    return l_Img

def cvDetectOneByClassifier( img, classifier, imgScale=1.2, minNeighbours=5 ):
    if not isinstance( classifier, cv2.CascadeClassifier ):
        raise TypeError( "Given classifier is not CascadeClassifier!" )
  
    l_DetectedArea = classifier.detectMultiScale( img, imgScale, minNeighbours )[0]

    return l_DetectedArea

def cvCropImgToArea( img, detectedArea, imgSize ):
    l_AreaX, l_AreaY, l_AreaWidth, l_AreaHeight = detectedArea

    l_CroppedImg = img[ l_AreaY:l_AreaY + l_AreaHeight, l_AreaX:l_AreaX + l_AreaWidth ]
    l_CroppedImg = cvResizeImage( l_CroppedImg, imgSize )

    return l_CroppedImg

def cvNormaliseImg( img ):
    l_NormalisedImg = img.astype( "float" ) / 255.0 #normalise

    return l_NormalisedImg

def cvExpandImgDimFromLeft( img ):
    l_ExpandedImg = np.expand_dims( img, axis=0 )

    return l_ExpandedImg

def cvExpandImgDimFromRight( img ):
    l_ExpandedImg = np.expand_dims( img, axis=-1 )

    return l_ExpandedImg