import tkinter as tk
import cv2
import numpy as np
import dlib
from imutils import face_utils
import winsound
from PIL import Image, ImageTk
import requests


account_sid = 'AC178a1873f4e94be40aa2c89c58f*****'
auth_token = '31b25862ad5a4970590b111626*****'
twilio_phone_number = '+18453793961'
emergency_contact = '+9163012*****'

def send_emergency_alert():
    # Send SMS
    url_sms = f'https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json'
    data_sms = {
        'From': twilio_phone_number,
        'To': emergency_contact,
        'Body': 'Driver detected in a dangerous state. Please check immediately!'
    }
    response_sms = requests.post(url_sms, data=data_sms, auth=(account_sid, auth_token))

    url_call = f'https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Calls.json'
    data_call = {
        'From': twilio_phone_number,
        'To': emergency_contact,
        'Url': 'http://demo.twilio.com/docs/voice.xml'
    }
    response_call = requests.post(url_call, data=data_call, auth=(account_sid, auth_token))

    if response_sms.status_code == 201 and response_call.status_code == 201:
        print('Emergency alert sent successfully!')
    else:
        print('Failed to send emergency alert.')

class EyeMouthStatusDetectionApp:
    def __init__(self, root, detector, predictor):
        self.root = root
        self.root.title("Eye and Mouth Status Detection")
        self.detector = detector
        self.predictor = predictor

        self.cap = cv2.VideoCapture(0)
        self.status_label = tk.Label(self.root, text="", font=("Helvetica", 16))
        self.status_label.pack()

        self.start_button = tk.Button(self.root, text="Start Detection", command=self.start_detection)
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(self.root, text="Stop Detection", command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.pack(pady=10)

        self.panel = tk.Label(self.root)
        self.panel.pack(padx=10, pady=10)

        self.sleep = 0
        self.drowsy = 0
        self.active = 0
        self.status = ""
        self.color = (0, 0, 0)
        self.is_detecting = False
        self.left_eye_state = []
        self.right_eye_state = []

    def start_detection(self):
        self.is_detecting = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.update_status()

    def stop_detection(self):
        self.is_detecting = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def compute(self, ptA, ptB):
        dist = np.linalg.norm(ptA - ptB)
        return dist

    def blinked(self, a, b, c, d, e, f):
        up = self.compute(b, d) + self.compute(c, e)
        down = self.compute(a, f)
        ratio = up / (2.0 * down)

        if ratio > 0.25:
            return 2
        elif 0.21 < ratio <= 0.25:
            return 1
        else:
            return 0

    def play_beep(self):
        frequency = 1500
        duration = 1000
        winsound.Beep(frequency, duration)

    def mouth_open(self, landmarks):
        top_lip = landmarks[50:53]
        top_lip = np.concatenate((top_lip, landmarks[61:64]))

        bottom_lip = landmarks[56:59]
        bottom_lip = np.concatenate((bottom_lip, landmarks[65:68]))

        top_mean = np.mean(top_lip, axis=0)
        bottom_mean = np.mean(bottom_lip, axis=0)

        distance = abs(top_mean[1] - bottom_mean[1])
        return distance

    def update_status(self):
        if not self.is_detecting:
            return

        _, frame = self.cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.detector(gray)

        for face in faces:
            landmarks = self.predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)

            left_blink = self.blinked(landmarks[36], landmarks[37],
                                       landmarks[38], landmarks[41], landmarks[40], landmarks[39])
            right_blink = self.blinked(landmarks[42], landmarks[43],
                                        landmarks[44], landmarks[47], landmarks[46], landmarks[45])
            mouth_status = self.mouth_open(landmarks)

            if left_blink == 0 or right_blink == 0:
                self.sleep += 1
                self.drowsy = 0
                self.active = 0
                self.left_eye_state.append(1)
                self.right_eye_state.append(1)
                if self.sleep > 6:
                    self.status = "SLEEPING !!!"
                    self.color = (255, 0, 0)
                    self.play_beep()
                    send_emergency_alert()  
            elif 0.21 < left_blink <= 0.25 or 0.21 < right_blink <= 0.25 or mouth_status > 20:
                self.sleep = 0
                self.active = 0
                self.drowsy += 1
                self.left_eye_state.append(0)
                self.right_eye_state.append(0)
                if self.drowsy > 6:
                    self.status = "Drowsy !"
                    self.color = (0, 0, 255)
            else:
                self.drowsy = 0
                self.sleep = 0
                self.active += 1
                self.left_eye_state.append(0)
                self.right_eye_state.append(0)
                if self.active > 6:
                    self.status = "Active :)"
                    self.color = (0, 255, 0)

        
        self.left_eye_state = self.left_eye_state[-10:]
        self.right_eye_state = self.right_eye_state[-10:]

       
        if sum(self.left_eye_state) > 5 or sum(self.right_eye_state) > 5:
            self.status = "Eye Closed"
            self.color = (255, 255, 255)

        cv2.putText(frame, self.status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.color, 3)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        self.panel.imgtk = imgtk
        self.panel.config(image=imgtk)
        self.panel.after(1, self.update_status)

    def run(self):
        self.root.mainloop()
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    app = EyeMouthStatusDetectionApp(root, detector, predictor)
    app.run()
