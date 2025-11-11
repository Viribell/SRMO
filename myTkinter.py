import tkinter as tk
from tkinter import filedialog

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