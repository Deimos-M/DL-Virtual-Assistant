import face_recognition
import cv2
import numpy as np
import pickle

import pyttsx3
import threading 

import speech_recognition as sr  
import os,sys
os.chdir(r'C:\Users\***\Desktop\web app\finalapp')
import nest_asyncio
nest_asyncio.apply()

class Facecamera(object):

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    new_frame = None

    #loading list
    with open('encod_list.data', 'rb') as filehandle:
        # read the data as binary data stream
        known_face_encodings = pickle.load(filehandle)
    with open('face_list.data', 'rb') as filehandle1:
        # read the data as binary data stream
        known_face_names = pickle.load(filehandle1)

    All_faces = [0,0,0,0,0,0]
    def second(self,name): 
        self.All_faces[self.All_faces.index(name)] = 0
        
    def __init__(self):
        self.video_capture = cv2.VideoCapture(0)

    def __del__(self):
        self.video_capture.release()  
     
    #to add new face
    def AddNewFace(self):
        if("Unknown" in self.All_faces):
            newadd_frame = self.new_frame[:, :, ::-1]
            newadd_frame = cv2.resize(newadd_frame, (0, 0), fx=3, fy=3)

            cv2.imwrite("NewPicture.jpg",newadd_frame)
                            
            engine = pyttsx3.init()
            engine.say("Adding new face..")
            engine.runAndWait()

            new = self.speech()
            global dst
            dst=None
            for file in os.listdir():
                src=file
                if src=='NewPicture.jpg':
                    #global dst
                    dst=new+".jpg"
                    os.rename(src,dst)
       
            new_image = face_recognition.load_image_file(dst)
            new_face_encoding = face_recognition.face_encodings(new_image)[0]
            self.known_face_encodings.append(new_face_encoding)
            self.known_face_names.append(new)

            engine = pyttsx3.init()
            engine.say("The face has been added succesfully.")
            engine.runAndWait()
            
        else :
            engine = pyttsx3.init()
            engine.say("No unknown face detected. ")
            engine.runAndWait()

        
    #speech to text
    def speech(self):
        # get audio from the microphone
        r = sr.Recognizer()
        
        engine = pyttsx3.init()
             
        with sr.Microphone() as source:
            engine.say("Please wait. Calibrating microphone...")
            engine.runAndWait() 
            # listen for 3 seconds and create the ambient noise energy level  
            r.adjust_for_ambient_noise(source, duration=3)
            
            engine.say(" please Speak the full name of the person")
            engine.runAndWait()
                                                                                   
            audio = r.listen(source)   

        try:
            name = r.recognize_google(audio)
            engine.say("You said " + name )
            engine.runAndWait()
            
        except sr.UnknownValueError:
            engine.say("Could not understand audio")
            engine.runAndWait()
            return self.speech()
        
        except sr.RequestError as e:
            engine.say("Check internet")
            engine.runAndWait()
            return self.speech()
        
        return name
    
    def Detection(self,flagrun):
        self.process_this_frame = True
        flag = 1
        
        if flagrun==1:
            engine = pyttsx3.init()
            engine.say('starting face recognition live stream.')
            engine.runAndWait()
            self.video_capture = cv2.VideoCapture(0)

        while (flag==1):
            # Grab a single frame of video
            ret, frame = self.video_capture.read()
            
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]
            
            self.new_frame = rgb_small_frame#copy
            
            # Only process every other frame of video to save time
            if self.process_this_frame:
                # Find all the faces and face encodings in the current frame of video
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"              

                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]: 
                        name = self.known_face_names[best_match_index]
                    self.face_names.append(name)
                    print(self.All_faces)
                    if name not in self.All_faces:
                        print(name)
                        print(self.All_faces)
                        if (self.All_faces[0] != 0):
                            for i in range(5,0,-1):
                                self.All_faces[i]=self.All_faces[i-1]
                        self.All_faces[0]=name

                        timer = threading.Timer(20, self.second,args=[name]) 
                        timer.start()

                        engine = pyttsx3.init()
                        engine.say('I see '+name)
                        engine.runAndWait()

            self.process_this_frame = not self.process_this_frame


            # Display the results
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            
            ret , jpeg = cv2.imencode('.jpg',frame)
            return jpeg.tobytes()

        
    #closing
    def close(self):
        #saving list ---------------how/need to save final unkown
        with open('encod_list.data', 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(self.known_face_encodings, filehandle)
        with open('face_list.data', 'wb') as filehandle1:
            # store the data as binary data stream
            pickle.dump(self.known_face_names, filehandle1)
            
        engine = pyttsx3.init()
        engine.say('stopping live stream.')
        engine.runAndWait()
        self.video_capture.release()
        
    
        
    