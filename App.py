from flask import Flask, render_template, Response, jsonify, redirect,url_for,request
from object_camera import VideoCamera
from facecam import Facecamera
import cv2

app = Flask(__name__)

video_stream = VideoCamera()
face_video_stream = Facecamera()


@app.route('/')
def index():
    return render_template('object_index.html')

def gen(camera):
    flag=1
    while True:
        frame = camera.get_frame(flag)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        flag=0

@app.route('/object_start_video')
def object_start_video():
        return Response(gen(video_stream),
                mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/object_close_video')
def object_close_video():
    video_stream.close()
    flag=1
    return render_template('object_index2.html')

###############connectivity

@app.route('/face_recog')
def face_recog():
    video_stream.switching()
    return render_template('face_index.html')
    
################face 

def gen_face(camera):
    flag=1
    while True:
        frame = camera.Detection(flag)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        flag=0

@app.route('/start_video')
def start_video():
    return Response(gen_face(face_video_stream),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/Add_face')
def Add_face():
    face_video_stream.AddNewFace()
    return redirect('/face_recog')
    
@app.route('/close_video')
def close_video():
    face_video_stream.close()
    flag=1
    return render_template('face_index2.html')


if __name__ == '__main__':
    app.run(threaded=False)