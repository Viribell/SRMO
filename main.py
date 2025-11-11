from myModel import *
from myUtils import *
from myCV import *

import tkinter as tk
from tkinter import filedialog

g_Model = None

g_DropoutValue = 0.5
g_ClassNum = 7
g_ReqImgSize = (48, 48)
g_SetBatchSize = 64
g_Epochs = 30

g_ModelsFolderName = "models"
g_ModelName = "myModel_save_full"
g_ModelExt = "keras"
g_TrainSetDir = "data/modelTraining"
g_TestSetDir = "data/modelTesting"
g_ClassNames = None

g_EmotionImage = "data/exampleImages/neutral.jpg"

#-----------------------------------------TK
class tkWidget:
    def __init__(self, widget):
        self.widget = widget

    def Pack(self, **kwargs): #pady, padx, side(top,bottom,left,right), expand(bool), fill(x,y,both)
        self.widget.pack( **kwargs )
        return self

    def PackPropagate(self, isOn):
        self.widget.pack_propagate( isOn )
        return self

    def Grid(self, **kwargs): #row, column, sticky(n,s,e,w), columnspan, rowspan, padx, pady, 
        self.widget.grid( **kwargs )
        return self

    def Place(self, **kwargs): #x, y,
        self.widget.place( **kwargs )
        return self

    def Background(self, color):
        self.widget.config( bg = color )
        return self

    def Bg(self, color):
    	return self.Background( color )

    def Foreground(self, color):
        self.widget.config( fg = color )
        return self

    def Fg(self, color):
    	return self.Foreground( color )

    def Image(self, img):
    	if img is None:
    		self.widget.config( image = "" )
    		self.widget.image = None
    	else:
    		self.widget.config( image = img )
    		self.widget.image = img
    		
    	return self

    def Text( self, content ):
    	self.widget.config( text = content )
    	return self

    def Font(self, textFont): #(name, size, type)
        self.widget.config( font = textFont )
        return self

    def Dimension(self, dimWidth, dimHeight):
        self.widget.config( width = dimWidth, height = dimHeight )
        return self

    def Config(self, **kwargs):
        self.widget.config( **kwargs )
        return self

    def Get(self):
        return self.widget

def tkCreateWindow( title, dimension, resizable = False ):
    l_Window = tk.Tk()
    l_Window.title( title )
    l_Window.geometry( dimension )
    l_Window.resizable( resizable, resizable )

    return l_Window

def tkAddFrame( parent, color="lightgray", dimension=(100, 100) ):
    l_Frame = tk.Frame( parent, bg = color, width = dimension[0], height = dimension[1] )

    return tkWidget( l_Frame )


def tkAddLabel( parent, content, textFont=("Arial", 12)  ):
    l_Label = tk.Label( parent, text = content, font = textFont )

    return tkWidget( l_Label )

def tkAddButton( parent, content, action, padding=8, textFont=("Arial", 12) ):
    l_Button = tk.Button( parent, text = content, command = action, font = textFont )

    return tkWidget( l_Button )

def tkOpenFileDialog( filterName, filterPattern ):
	l_Path = filedialog.askopenfilename(
    	filetypes=[(filterName, filterPattern)]
	)	

	return l_Path

#-----------------------------------------MAIN_LOWER_FUNC
def GetNormalisedEmotion( imgPath, imgSize ):
    l_Img = cvLoadImage( imgPath )
    l_Img = cvConvertImageToGrayscale( l_Img )

    l_Face = cvDetectOneByClassifier( l_Img, cvGetCascadeClassifier( g_cvFaceClassifierName ) )

    if len(l_Face) == 0:
        print( "Face not detected." )
        return

    l_Emotion = cvCropImgToArea( l_Img, l_Face, imgSize )
    l_Emotion = cvNormaliseImg( l_Emotion )
    l_Emotion = cvExpandImgDimFromLeft( l_Emotion ) #batch
    l_Emotion = cvExpandImgDimFromRight( l_Emotion ) #channel

    return l_Emotion

def PredictEmotion( emotionImg ):
    l_Predictions = g_Model.predict( emotionImg )

    return l_Predictions

def CategorizeEmotion( predictions ):
    l_ClassIndex = np.argmax( predictions )
    l_ClassName = GetFolderByValueFromDict( g_ClassNames, l_ClassIndex )

    return l_ClassName

def CreateAndTrainNewModel():
    global g_Model, g_ClassNames

    g_Model = GetLearningModel();
    g_Model.summary()

    l_TrainItr = GetTrainingIterator( g_TrainSetDir, g_ReqImgSize, g_SetBatchSize )
    l_TestItr = GetTestingIterator( g_TestSetDir, g_ReqImgSize, g_SetBatchSize )

    l_History = TrainModel( g_Epochs, g_Model, l_TrainItr, l_TestItr )

    SaveModel( g_Model, g_ModelName )

    g_ClassNames = GetClassDict( l_TrainItr )

#-----------------------------------------MAIN_UPPER_FUNC
def InitSystem():
    global g_Model, g_ClassNames

    l_Path = GetFullModelPath( g_ModelsFolderName, g_ModelName, g_ModelExt )

    if not os.path.exists( l_Path ):
        print( "Creating Fresh Model\n" )
        CreateAndTrainNewModel()
    else:
        print( "Loading Existing Model\n" )
        g_Model = LoadModel( g_ModelName )
        g_Model.summary()
        g_ClassNames = GetFoldersDict( g_TrainSetDir )

    print( "\n\n\n" )

def ProcessImageForEmotion( imgPath ):
    l_Emotion = GetNormalisedEmotion( imgPath, g_ReqImgSize )
    l_Predictions = PredictEmotion( l_Emotion )
    l_ClassName = CategorizeEmotion( l_Predictions )

    return l_ClassName

#-----------------------------------------MAIN

InitSystem()