import subprocess
import threading
import time
import tkinter
import customtkinter  # <- import the CustomTkinter module
import tkinter as tk
from tkinter import ttk

from tkinter import filedialog, HORIZONTAL, DISABLED, NORMAL
from tkinter.ttk import Progressbar
import os
from datetime import datetime, date
import numpy as np
from PIL import ImageTk,Image
import subprocess
import time
from tkVideoPlayer import TkinterVideo
import pandas as pd
import glob
import os


idno="1"

fno="15"

videopath = "output_video.mp4";


def seekForward():
    videoplayer.seek(int(videoplayer.current_duration()+5))

def seekBack():
    videoplayer.seek(int(videoplayer.current_duration()-5))


customtkinter.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"

#Replace <command=example> with <command=threading.Thread(target=example).start>

root_tk = customtkinter.CTk()
root_tk.geometry("1250x700")
root_tk.title("Multiple Object Tracking")



frame_1 = customtkinter.CTkFrame(master=root_tk, corner_radius=15)
frame_1.pack(pady=10, padx=2.5, fill="both", expand=True,side="left")


label_1 = customtkinter.CTkLabel(master=frame_1,text="MULTI OBJECT TRACKER",width=220,height=55,fg_color="Grey",corner_radius=15,font=("Calibri",32))
label_1.pack(pady=13, padx=30)

videoplayer = TkinterVideo(master=frame_1,scaled=True)
videoplayer.load(videopath)
videoplayer.pack(ipadx=350, ipady=250,fill="both",padx=60,pady=60)
videoplayer.play()



frame_2 = customtkinter.CTkFrame(master=root_tk, corner_radius=15)
#frame_2.pack(pady=10, padx=2.5, fill="both", expand=True,side="right")




pause_button = customtkinter.CTkButton(master=frame_1, corner_radius=8,text="Pause", command=videoplayer.pause,state=NORMAL,width=30)
pause_button.pack(pady=13, padx=30)
pause_button.place(relx=0.055,rely=0.92)

play_button = customtkinter.CTkButton(master=frame_1, corner_radius=8,text="Play", command=videoplayer.play,state=NORMAL,width=30)
play_button.pack(pady=13, padx=30)
play_button.place(relx=0.118,rely=0.92)

seek_forward = customtkinter.CTkButton(master=frame_1, corner_radius=8,text="Forward", command=seekForward,state=NORMAL,width=30)
seek_forward.pack(pady=13, padx=30)
seek_forward.place(relx=0.17,rely=0.92)

seek_backward = customtkinter.CTkButton(master=frame_1, corner_radius=8,text="Backward", command=seekBack,state=NORMAL,width=30)
seek_backward.pack(pady=13, padx=30)
seek_backward.place(relx=0.245,rely=0.92)



container = ttk.Frame(frame_2)
canvas = tk.Canvas(container)
scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
scrollable_frame = ttk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

canvas.configure(yscrollcommand=scrollbar.set,width=135,height=600)

frame_2.pack(pady=110, padx=2.5, fill="both", expand=True,side="right")
container.pack()
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")


with open('track_ids.txt', 'r') as f:
    tio = f.readlines()
with open('track_hits.txt', 'r') as f:
    tho = f.readlines()


    for x in range(len(tio)):
        tid = tio[x][:-1]
        th = tho[x][:-1]
        img = Image.open("./images/"+tid+".jpg")
        img = img.resize((130, 100))
        img_tk = ImageTk.PhotoImage(img)
        img_label = customtkinter.CTkLabel(scrollable_frame, image=img_tk, text="")
        img_label.pack(pady=10)
        img_label.place()
        id_label = customtkinter.CTkLabel(master=scrollable_frame, text="ID Number: " + tid,text_color="black")
        id_label.pack()
        frames_label = customtkinter.CTkLabel(master=scrollable_frame, text="Frames: " + th,text_color="black")
        frames_label.pack()


#root_tk.after(get_vals())

root_tk.mainloop()


