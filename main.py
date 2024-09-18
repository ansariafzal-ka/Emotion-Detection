import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model

class EmotionDetector:
    
    def __init__(self):
        print("running...")
        self.window = tk.Tk()
        self.window.geometry("600x600")
        self.window.title("Emotion Detector Model")
        self.window.resizable(False, False)
        
        self.cap = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.model = load_model('emotion_detector.h5')
        self.emotion_labels = ["angry", "happy", "neutral"]
        
        self.label = tk.Label(self.window)
        self.label.pack(padx=10, pady=10)

        self.update_frame()

        self.window.mainloop()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (48, 48))
                face = face.astype('float32') / 255
                face = np.expand_dims(face, axis=-1)
                face = np.expand_dims(face, axis=0)

                emotion_pred = self.model.predict(face)
                emotion_label = self.emotion_labels[np.argmax(emotion_pred)]

                cv2.rectangle(frame, (x, y), (x+w, y+h), (225, 225, 225), 2)
                label_position = (x, y - 10)
                cv2.putText(frame, emotion_label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)

            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)

        self.window.after(10, self.update_frame)


if __name__ == "__main__":
    emotion_detector = EmotionDetector()
