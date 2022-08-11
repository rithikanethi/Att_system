import tkinter as tk 
from tkinter import Message, Text
from tkinter import*
import cv2
import os
import numpy as np
import sqlite3  
import datetime 
import time 
import tkinter.ttk as ttk 
import tkinter.font as font 
from pathlib import Path
from PIL import Image


window =tk.Tk() 
window.title("Face_Recognizer") 
window.configure(background ='snow') 
window.grid_rowconfigure(0, weight = 1) 
window.grid_columnconfigure(0, weight = 1) 
message = tk.Label( 
	window, text ="Face-Recognition-System", 
	bg ="#00008C", fg = "white", width = 50, 
	height = 3, font = ('times', 30, 'bold')) 
	
message.place(x = 100, y = 20) 

img=PhotoImage(file=r"C:\Users\Veera Kumar\Desktop\Attendance_system_mini\cummins1.png")
tt=tk.Label(window,image=img)
tt.place(x=650,y=250)

lbl = tk.Label(window, text = "Id", 
width = 20, height = 2, fg ="black", 
bg = "white", font = ('times', 25, ' bold ') ) 
lbl.place(x = 350, y = 400)

txt= tk.Entry(window, 
width = 20, bg ="white", 
fg ="black", font = ('times', 25, ' bold ')) 
txt.place(x = 600, y = 415) 

lbl2 = tk.Label(window, text ="Name", 
width = 20, fg ="black", bg ="white", 
height = 2, font =('times', 25, ' bold ')) 
lbl2.place(x = 350, y = 500)


txt2 = tk.Entry(window, width = 20, 
bg ="white", fg ="black", 
font = ('times', 25, ' bold ') ) 
txt2.place(x = 600, y = 515) 


def TakeImages():
    conn=sqlite3.connect('facedatabasedb.sqlite')
    cur=conn.cursor()
    cur.executescript(''' CREATE TABLE IF NOT EXISTS Data (
                      ID INTEGER NOT NULL,
                      Name TEXT NOT NULL);

                      CREATE TABLE IF NOT EXISTS Detected (
                      Name TEXT NOT NULL,
                      Time TEXT NOT NULL) ''')
    Id=(txt.get())
    Name=(txt2.get())
    cap=cv2.VideoCapture(0)
    faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cur.execute('INSERT INTO Data (ID,Name) VALUES (?,?)',(Id,Name))
    conn.commit()
    cur.close()
    sampleNum=0
    while True:
        ret,frame=cap.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces =faceCascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            sampleNum=sampleNum+1
            cv2.imwrite("database/User."+str(Id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
            cv2.imshow("Video",frame)
        cv2.waitKey(100) 
        if (sampleNum>20):
            break
    cap.release()
    cv2.destroyAllWindows()
    res = ("Images Saved for ID : "+Id +" Name : "+ Name)
    message.configure(text = res) 

def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    path='database'
    def getImagesWithID(path):
        imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
        faces=[]
        labels=[]
        for imagePath in imagePaths:
            faceImg=Image.open(imagePath).convert('L')
            faceNp=np.array(faceImg,'uint8')
            label=int(os.path.split(imagePath)[-1].split('.')[1]) 
            faces.append(faceNp)
            print(label)
            labels.append(label)
            cv2.imshow("training",faceNp)
            cv2.waitKey(10)
        return labels,faces
    labels,faces=getImagesWithID(path)
    recognizer.train(faces,np.array(labels))
    recognizer.save('recognizer/trainingData.yml')
    cv2.destroyAllWindows()
    res = "Image Trained"
    message.configure(text = res)

def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("recognizer\\trainingData.yml")
    faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
    path='database'
    def getProfile(Id):
        conn=sqlite3.connect('facedatabasedb.sqlite')
        cursor=conn.execute('SELECT * FROM Data WHERE ID=?',str(Id))
        profile=None
        for row in cursor:
            profile=row
            conn.commit()
            conn.close()
            return profile
    cap = cv2.VideoCapture(0) 
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        ret,frame =cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for(x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (225, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            profile=getProfile(Id)
            timet=datetime.datetime.now()
            tt=timet.strftime("%d %m %Y %H:%M")
            if(conf < 75):
                if profile!=None:
                    cv2.putText(frame,str(profile[1]),(x+30,y+h),font,1,(0,0,255),2)
                    cv2.putText(frame,str(tt),(x+60,y+h+60),font,1,(0,0,255),2)
                    cv2.putText(frame,str(conf),(x+90,y+h+90),font,1,(0,0,255),2)
                    conn=sqlite3.connect('facedatabasedb.sqlite')
                    cur=conn.cursor()
                    cur.execute('INSERT INTO Detected (Time,Name) VALUES (?,?)',(tt,profile[1]))
                    conn.commit()
                    conn.close()
            else:
                if profile!=None:
                    cv2.putText(frame,"Unknown",(x+30,y+h),font,1,(0,0,255),2)
                    cv2.putText(frame,str(tt),(x+60,y+h+60),font,1,(0,0,255),2)
                    cv2.putText(frame,str(conf),(x+90,y+h+90),font,1,(0,0,255),2)
                    conn=sqlite3.connect('facedatabasedb.sqlite')
                    cur=conn.cursor()
                    cur.execute('INSERT INTO Detected (Time,Name) VALUES (?,?)',(tt,"Unknown"))
                    conn.commit()
                    conn.close()
        cv2.imshow('Video', frame)
        if (cv2.waitKey(1)== ord('q')):
            break
    cap.release()
    cv2.destroyAllWindows() 
    
takeImg = tk.Button(window, text ="Add Student", 
command = TakeImages, fg ="black", bg ="#BCD2E8", 
width = 20, height = 3, activebackground = "grey", 
font =('times', 15, ' bold ')) 
takeImg.place(x = 100, y = 600) 
trainImg = tk.Button(window, text ="Train Images", 
command = TrainImages, fg ="black", bg ="#BCD2E8", 
width = 20, height = 3, activebackground = "grey", 
font =('times', 15, ' bold ')) 
trainImg.place(x = 400, y = 600) 
trackImg = tk.Button(window, text ="Attendance", 
command = TrackImages, fg ="black", bg ="#BCD2E8", 
width = 20, height = 3, activebackground = "grey", 
font =('times', 15, ' bold ')) 
trackImg.place(x = 700, y = 600) 
quitWindow = tk.Button(window, text ="Quit", 
command = window.destroy, fg ="black", bg ="#BCD2E8", 
width = 20, height = 3, activebackground = "grey", 
font =('times', 15, ' bold ')) 
quitWindow.place(x = 1000, y = 600) 


window.mainloop() 
    
