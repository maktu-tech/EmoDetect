
import numpy as np
import pyautogui as SS
import time
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, Activation, Dropout, MaxPooling2D, Flatten, Resizing, Lambda
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import mixed_precision as m_p
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image as Kimage
from PIL import Image as Pimage


import wx
from wx.core import STAY_ON_TOP


input_size=(48,48,1)
classes = 7
model = tf.keras.Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
model.load_weights('model_bestweight.h5')


emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Feared", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

def predict(cord1, cord2):

    l1=[0]*7
    image = SS.screenshot()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    image = image[ cord1[1]:cord2[1],cord1[0]:cord2[0]] 

    # Facedetect
    face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    all_faces = face_detect.detectMultiScale(image, minNeighbors=10) 

    for (x,y,w,h) in all_faces:
        img = image[y:y+h, x:x+w]
        cropped_img = np.expand_dims(np.expand_dims(
                cv2.resize(img, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        index = int(np.argmax(prediction))
        print("emotion: ",emotion_dict[index])
        l1[index]+=1
    return l1.index(max(l1))

import time

class ResultFrm(wx.Frame):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

        self.Bind(wx.EVT_CLOSE,self.onclose)
        self.panel = wx.Panel(self)
        self.lbl = wx.StaticText(self.panel,label = "Emotion Here",pos = (0,20),size=(300,50), style = wx.ALIGN_CENTER_HORIZONTAL)
        self.lbl.SetFont(wx.Font(15, wx.DECORATIVE,wx.NORMAL,wx.NORMAL))
        self.timer = wx.Timer(self, wx.ID_ANY)
        self.timer.Start(1000)
        self.Bind(wx.EVT_TIMER, self.firepred)
        self.dict = {0:0,1:0,2:0,3:0,4:0,5:0,6:0}
        self.lst = []

    def firepred(self, evt):
        if(self.pfr.startP):
            ind = predict(self.pfr.c1,self.pfr.c2)
            self.dict[ind] += 1
            self.lst.append(ind)
            ind_emo = max(zip(self.dict.values(), self.dict.keys()))[1]
            self.lbl.SetLabel(emotion_dict[ind_emo])       
            self.Refresh()

        if(len(self.lst) > 10):
            self.dict[self.lst[0]] -= 1
            self.lst.pop(0) 
        
    def onclose(self,evt):
        self.pfr.Destroy()
        self.Destroy()

class SelectableFrame(wx.Frame):

    c1 = None
    c2 = None
    

    def __init__(self, parent=None, id=-1, title=""):
        wx.Frame.__init__(self, parent, id, title, size=wx.DisplaySize())

        self.panel = wx.Panel(self, size=self.GetSize())

        self.panel.Bind(wx.EVT_MOTION, self.OnMouseMove)
        self.panel.Bind(wx.EVT_LEFT_DOWN, self.OnMouseDown)
        self.panel.Bind(wx.EVT_LEFT_UP, self.OnMouseUp)
        self.panel.Bind(wx.EVT_PAINT, self.OnPaint)
        self.startP = False

        

    def OnMouseMove(self, event):
        if event.Dragging() and event.LeftIsDown():
            self.c2 = event.GetPosition()
            self.Refresh()

    def OnMouseDown(self, event):
        self.c1 = event.GetPosition()

    def OnMouseUp(self, event):
        self.Hide()
        self.nfr.Show(True)
        self.startP = True 

    def OnPaint(self, event):
        if self.c1 is None or self.c2 is None: return

        dc = wx.PaintDC(self.panel)
        dc.SetPen(wx.Pen('green', 10))
        # wx.Colour
        dc.SetBrush(wx.Brush(wx.Colour(0, 100, 0), wx.TRANSPARENT))

        dc.DrawRectangle(self.c1.x, self.c1.y, self.c2.x - self.c1.x, self.c2.y - self.c1.y)


    def PrintPosition(self, pos):
        return str(pos.x) + " " + str(pos.y)

class MyApp(wx.App):

    def OnInit(self):
        frame = SelectableFrame()
        frame2 = ResultFrm(None,title ="frm2",size = (300,100),style= wx.DEFAULT_FRAME_STYLE | wx.STAY_ON_TOP)
        frame.nfr = frame2
        frame2.pfr = frame
        frame.Show(True)
        frame.SetTransparent(50)
        self.SetTopWindow(frame)

        return True


app = MyApp(0)
app.MainLoop()