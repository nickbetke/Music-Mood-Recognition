import tkinter as tk     # python 3
from tkinter import *
from tkinter.ttk import *
from tkinter import font as tkFont
from tkinter.filedialog import askopenfilename
from tkinter import messagebox

import numpy as np
from pydub import AudioSegment as pas
import librosa as lr
import joblib

global do


class SampleApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.title_font = tkFont.Font(family='Helvetica', size=20, weight="bold", slant="italic")
        self.mood_font = tkFont.Font(family='Helvetica', size=16, slant="italic")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (StartPage, PageOne):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")


    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()


class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent,bg = "crimson")
        self.controller = controller
        label = tk.Label(self, text="Music Mood Recogniser", font=controller.title_font)
        label.pack(side="top", fill="x", pady=20)
#for sending filename to other page
        self.filename = ""

        self.lbl = tk.Label(self, text="Select an Audio file", font=controller.mood_font, bg = "bisque", fg="black")
        self.lbl.pack(side="top", fill="x", pady=200)
        #==================================================
        # style = Style()
        # style.configure('TButton', font = 
        #        ('calibri', 20, 'bold'), 
        #             borderwidth = '4') 
        # style.configure('TButton', font = 
        #        ('calibri', 10, 'bold', 'underline'), 
        #         foreground = 'red')
        #================================================   
        self.button1 = tk.Button(self, text="Browse", bg="greenyellow", command=self.clicked)
        self.button1.place(x =250, y = 400)
        
        self.button2 = tk.Button(self, text="Next", bg="chartreuse",command=lambda: controller.show_frame("PageOne"))
        self.button2["state"] = "disabled"
        self.button2.place(x =400, y = 400)

    def lop(self):
        PageOne.predict()

    def clicked(self):
        global do
        self.filename = askopenfilename(title = "Select an audio file", filetypes = (("mp3 files", "*.mp3*"), 
                                                                                ("m4a files", "*.m4a*"),
                                                                                ("All files", "*.*")))
        if self.filename:
            do.set(self.filename)
            self.button2["state"] = "normal"
            self.lbl.configure(text=self.filename)
        print(self.filename)
        
#Frame 2
class PageOne(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg = "crimson")
        self.controller = controller

        # global do
        # k1 = do.get()

        label = tk.Label(self, text="Music Mood Recogniser", font=controller.title_font)
        label.pack(side="top", fill="x", pady=20)

        button = tk.Button(self, text="Go Back", bg="chartreuse", command=lambda: controller.show_frame("StartPage"))
        button.pack()

        button2 = tk.Button(self, text="Recognize", bg="chartreuse", command = self.testg)
        button2.pack()

        #mood Label
        
        label2 = tk.Label(self, text="mood", font=controller.title_font, bg = "cornsilk", fg = "black")
        label2.pack(side="top", fill="x", pady=50)
        # label2.place(x=100 ,y=250)

        #self.predict()



    def testg(self):
        global do
        f1 = do.get()
        print("mooded")
        print(f1)
        self.predict()

    def predict(self):
        global do
        ad = do.get()
        n = ad.split('/')
        i = 'cutpiece.mp3'
        i = n[-1]
        if 'mp3' in i:
            song = pas.from_mp3(ad)
        if 'm4a' in i:
            song = pas.from_file(ad)
        ii = i[:-3] + 'wav'
        op = song[:30000]
        op.export('/home/nick/1PROjectX/trash/' + ii, format = 'wav')

        feat1 = np.empty((0,181))
        audio, freq = lr.load('/trash/' + ii)
        stft = np.abs(lr.stft(audio))
        mfcc = np.mean(lr.feature.mfcc(y = audio, sr = freq, n_mfcc=40).T, axis=0)
        mel = np.mean(lr.feature.melspectrogram(audio, sr = freq).T, axis=0)
        contrast = np.mean(lr.feature.spectral_contrast(S = stft, sr = freq).T, axis=0)
        tonnetz = np.mean(lr.feature.tonnetz(y = lr.effects.harmonic(audio), sr = freq).T, axis=0)
        ext_feat = np.hstack([mfcc, mel, contrast, tonnetz])
        feat1 = np.vstack([feat1, ext_feat])

        filename = '/46/decison-tree-model36.sav'
        x = joblib.load(filename)
        ww = x.predict(feat1)[0]
        print(x.predict(feat1)[0])
        label3 = tk.Label(self, text="DTree: " + str(ww), font='Helvetica 12 bold', bg = "lightsteelblue4", fg = "white")
        label3.pack(side="top", fill="x", pady=20)

        filename = '/46/linear-svm-model36.sav'
        x = joblib.load(filename)
        ww = x.predict(feat1)[0]
        label4 = tk.Label(self, text="SVM: " + str(ww), font='Helvetica 12 bold',  bg = "lightsteelblue4", fg = "white")
        label4.pack(side="top", fill="x", pady=20)

        filename = '/46/naive-bayes-model36.sav'
        x = joblib.load(filename)
        ww = x.predict(feat1)[0]
        label5 = tk.Label(self, text="NaiveBayes: " + str(ww), font='Helvetica 12 bold', bg = "lightsteelblue4", fg = "white")
        label5.pack(side="top", fill="x", pady=20)

        filename = '/46/random-forest-model36.sav'
        x = joblib.load(filename)
        ww = x.predict(feat1)[0]
        label6 = tk.Label(self, text="Randomforest: " + str(ww), font='Helvetica 12 bold', bg = "lightsteelblue4", fg = "white")
        label6.pack(side="top", fill="x", pady=20)
        # label2['text'] = 'mood: ' + str(ww)
        


if __name__ == "__main__":
    app = SampleApp()
    app.title("Music Mood Recogniser")
    app.geometry("800x720+0+0")
    app.resizable(True, True)
    do = StringVar()
    app.mainloop()