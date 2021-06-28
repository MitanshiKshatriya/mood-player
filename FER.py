from keras.models import model_from_json # loading the model saved in json file
import numpy as np
import cv2
import argparse
import os
from spotify_strike_final import play_song

class FacialExpressionModel(object):
    # EMOTIONS_LISTT = ["Angry", "Disgust", "Fear", "Happy", "Sad",
    #                  "Surprise", "Neutral"]
    EMOTIONS_LIST = ["Happy","Sad","Neutral"]

    def __init__(self, model_json_file, model_weights_file):
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read() # loaidng the model
            self.loaded_model = model_from_json(loaded_model_json)

        self.loaded_model.load_weights(model_weights_file)
        print("Model loaded from disk")
        self.loaded_model.summary()

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img) #[0.9,0.8...]
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)] #[0.0 1.0 0.0] 1


parser = argparse.ArgumentParser()
parser.add_argument("source") #python fer.py source fps webcam 25
parser.add_argument("fps")
args = parser.parse_args()
cap = cv2.VideoCapture(os.path.abspath(args.source) if not args.source == 'webcam' else 0)
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # detect faces in image
font = cv2.FONT_HERSHEY_SIMPLEX
cap.set(cv2.CAP_PROP_FPS, int(args.fps))


def getdata():
    success, fr = cap.read()
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    return faces, fr, gray


def start_app(cnn):
    mood=''
    ctr=0
    while cap.isOpened():
        faces, fr, gray_fr = getdata()
        for (x, y, w, h) in faces:
            fc = gray_fr[y:y + h, x:x + w] #face
            roi = cv2.resize(fc, (48, 48)) #input size
            pred = cnn.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
            mood=pred
            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 1) 
            cv2.rectangle(fr, (x, y), (x + w, y + h), (255, 0, 0), 1)
            cv2.putText(fr,'press space to play your mood or wait for 10 secs',  
                (0, 20),  
                font, 1,  
                (0, 255, 255),  
                2,  
                cv2.LINE_4) 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(1) & 0xFF == ord(' '):
            if(mood!=''):
                print("playing for mood = ",mood)
                play_song(mood)
        cv2.imshow('Facial Emotion Recognition', fr)
   
    cap.release()
    cv2.destroyAllWindows()

def writesome(pred):
    print(pred)

    

if __name__ == '__main__':
    model = FacialExpressionModel("model554.json", "weights554.h5")
    start_app(model)
