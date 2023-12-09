from flask import Flask, render_template, Response, jsonify, stream_with_context, redirect, url_for, request
import cv2
import mediapipe as mp
import math
import cv2
import numpy as np
import cvzone
from cvzone.PoseModule import PoseDetector
import threading
import time
import winsound


app = Flask(__name__)

video_access_event_pushup = threading.Event()
video_access_event_pushup.set()

video_access_event_squat = threading.Event()
video_access_event_squat.set()

#variables-------
counterp = 0
directionp = 0
pd_pushup = PoseDetector(trackCon=0.70, detectionCon=0.70)
pd_squat = PoseDetector(trackCon=0.70, detectionCon=0.70)
cap_pushup = cv2.VideoCapture(cv2.CAP_V4L2)
cap_squat = cv2.VideoCapture(cv2.CAP_V4L2)
if cap_pushup.isOpened():
    pass
import time
 # Import the pygame library
import os


#----------------------------------------------
#Push Up counter
#Cv code

def anglesp(lmlist,p1,p2,p3,p4,p5,p6,drawpoints):
        global img
        global counterp
        global directionp

        if len(lmlist)!= 0:
            point1 = lmlist[p1]
            point2 = lmlist[p2]
            point3 = lmlist[p3]
            point4 = lmlist[p4]
            point5 = lmlist[p5]
            point6 = lmlist[p6]

            x1, y1, _ = point1
            x2, y2, _ = point2
            x3, y3, _ = point3
            x4, y4, _ = point4
            x5, y5, _ = point5
            x6, y6, _ = point6

            if drawpoints == True:
                cv2.circle(img,(x1,y1),10,(255,0,255),5)
                cv2.circle(img, (x1, y1), 15, (0,255, 0),5)
                cv2.circle(img, (x2, y2), 10, (255, 0, 255), 5)
                cv2.circle(img, (x2, y2), 15, (0, 255, 0), 5)
                cv2.circle(img, (x3, y3), 10, (255, 0, 255), 5)
                cv2.circle(img, (x3, y3), 15, (0, 255, 0), 5)
                cv2.circle(img, (x4, y4), 10, (255, 0, 255), 5)
                cv2.circle(img, (x4, y4), 15, (0, 255, 0), 5)
                cv2.circle(img, (x5, y5), 10, (255, 0, 255), 5)
                cv2.circle(img, (x5, y5), 15, (0, 255, 0), 5)
                cv2.circle(img, (x6, y6), 10, (255, 0, 255), 5)
                cv2.circle(img, (x6, y6), 15, (0, 255, 0), 5)

                cv2.line(img,(x1,y1),(x2,y2),(0,0,255),6)
                cv2.line(img, (x2,y2), (x3, y3), (0, 0, 255), 6)
                cv2.line(img, (x4, y4), (x5, y5), (0, 0, 255), 6)
                cv2.line(img, (x5, y5), (x6, y6), (0, 0, 255), 6)
                cv2.line(img, (x1, y1), (x4, y4), (0, 0, 255), 6)

            lefthandangle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                                         math.atan2(y1 - y2, x1 - x2))

            righthandangle = math.degrees(math.atan2(y6 - y5, x6 - x5) -
                                          math.atan2(y4 - y5, x4 - x5))

            # print(lefthandangle,righthandangle)

            leftHandAngle = int(np.interp(lefthandangle, [-30, 180], [100, 0]))
            rightHandAngle = int(np.interp(righthandangle, [34, 173], [100, 0]))

            left, right = leftHandAngle, rightHandAngle

            if  right >= 100:
                if directionp == 0:
                    counterp += 0.5
                    directionp = 1
                    
            if  right <= 90:
                if directionp == 1:
                    counterp += 0.5
                    directionp = 0


            cv2.rectangle(img, (0, 0), (120, 120), (255, 0, 0), -1)
            cv2.putText(img, str(int(counterp)), (20, 70), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.6, (0, 0, 255), 7)

            leftval  = np.interp(right,[0,100],[400,200])
            rightval = np.interp(right, [0, 100], [400, 200])

         


            if left > 70:
                cv2.rectangle(img, (952, int(leftval)), (995, 400), (0, 0, 255), -1)

            if right > 70:
                cv2.rectangle(img, (8, int(leftval)), (50, 400), (0, 0, 255), -1)

#pushup-counter function
def process_videop():
    global cap_pushup
    global pd_pushup
    global img
    global counterp
    global directionp
    global video_access_event_pushup

    while video_access_event_pushup.is_set():
        ret, frame =cap_pushup.read()

        if ret:
            # Flip the frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)

            # Convert the BGR image to RGB
           # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with MediaPipe Pose
            img = cv2.resize(frame, (1000, 500))
            cvzone.putTextRect(img, 'AI Push Up Counter', [345, 30], thickness=2, border=2, scale=2.5)
            pd_pushup.findPose(img, draw=0)
            lmlist, bbox = pd_pushup.findPosition(img, draw=0, bboxWithHands=0)

            anglesp(lmlist, 11, 13, 15, 12, 14, 16, drawpoints=1)

            # Display the resulting frame
            _, jpeg = cv2.imencode('.jpg', img)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

    cap_pushup.release()
    cv2.destroyAllWindows()


#------------------------------------------------

#Squat counter
counters = 0
directions = 0
class angleFinder:
    def __init__(self,lmlist,p1,p2,p3,p4,p5,p6,drawPoints):
        self.lmlist = lmlist
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.p5 = p5
        self.p6 = p6
        self.drawPoints = drawPoints
    #    finding angles

    def angles(self):
        if self.lmlist is not None and len(self.lmlist) != 0:
            point1 = self.lmlist[self.p1]
            point2 = self.lmlist[self.p2]
            point3 = self.lmlist[self.p3]
            point4 = self.lmlist[self.p4]
            point5 = self.lmlist[self.p5]
            point6 = self.lmlist[self.p6]

            x1, y1, _ = point1
            x2, y2, _ = point2
            x3, y3, _ = point3
            x4, y4, _ = point4
            x5, y5, _ = point5  
            x6, y6, _ = point6

            # calculating angle for left and right hands
            leftHandAngle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                                        math.atan2(y1 - y2, x1 - x2))
            

            leftHandAngle = int(np.interp(leftHandAngle, [42, 143], [100, 0]))
          

            # drawing circles and lines on selected points
            if self.drawPoints == True:
                cv2.circle(imgs, (x1, y1), 10, (0, 255, 255), 5)
                cv2.circle(imgs, (x1, y1), 15, (0, 255, 0), 6)
                cv2.circle(imgs, (x2, y2), 10, (0, 255, 255), 5)
                cv2.circle(imgs, (x2, y2), 15, (0, 255, 0), 6)
                cv2.circle(imgs, (x3, y3), 10, (0, 255, 255), 5)
                cv2.circle(imgs, (x3, y3), 15, (0, 255, 0), 6)
                cv2.circle(imgs, (x4, y4), 10, (0, 255, 255), 5)
                cv2.circle(imgs, (x4, y4), 15, (0, 255, 0), 6)
                cv2.circle(imgs, (x5, y5), 10, (0, 255, 255), 5)
                cv2.circle(imgs, (x5, y5), 15, (0, 255, 0), 6)
                cv2.circle(imgs, (x6, y6), 10, (0, 255, 255), 5)
                cv2.circle(imgs, (x6, y6), 15, (0, 255, 0), 6)

                cv2.line(imgs,(x1,y1),(x2,y2),(0,0,255),4)
                cv2.line(imgs, (x2, y2), (x3, y3), (0, 0, 255), 4)
                cv2.line(imgs, (x4, y4), (x5, y5), (0, 0, 255), 4)
                cv2.line(imgs, (x5, y5), (x6, y6), (0, 0, 255), 4)
                cv2.line(imgs, (x1, y1), (x4, y4), (0, 0, 255), 4)

            return (leftHandAngle)
        else:
            return 0

# Function to process video frames for squat
def process_videos():
    global cap_squat
    global pd_squat
    global imgs
    global counters
    global directions
    global video_access_event_squat

    while video_access_event_squat.is_set():
        ret, frame = cap_squat.read()

        if ret:
            # Flip the frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)

            imgs = cv2.resize(frame, (1000, 500))
            cvzone.putTextRect(imgs, 'AI Squats Counter', [345, 30], thickness=2, border=2, scale=2.5)
            pd_squat.findPose(imgs, draw=0)
            lmList, bbox = pd_squat.findPosition(imgs, draw=0, bboxWithHands=0)

            angle1 = angleFinder(lmList, 24, 26, 28, 23, 25, 27, drawPoints=True)
            left = angle1.angles()

            # Counting number of shoulder ups
            if left >= 90:
                
                if directions == 0:
                    counters += 0.5
                    directions = 1
            if left <= 70:
            
                if directions == 1:
                    counters += 0.5
                    directions = 0

            cv2.rectangle(imgs, (0, 0), (120, 120), (255, 0, 0), -1)
            cv2.putText(imgs, str(int(counters)), (1, 70), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.6, (0, 0, 255), 6)

            # Display the resulting frame
            _, jpegs = cv2.imencode('.jpg', imgs)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpegs.tobytes() + b'\r\n\r\n')

    cap_squat.release()
    cv2.destroyAllWindows()

NEWS_API_KEY = 'b8b52943d64b4eeca0afb40b47c23035'
#-----------------------------------------------------------
#News page
from flask import Flask, render_template
import requests



# Replace 'YOUR_API_KEY' with your actual NewsAPI key
NEWS_API_KEY = 'b8b52943d64b4eeca0afb40b47c23035'
NEWS_API_ENDPOINT = 'https://newsapi.org/v2/top-headlines'

def get_cricket_news():
    params = {
        'apiKey': NEWS_API_KEY,
        'q': 'cricket',
        'category': 'sports',
        'pageSize': 10,
    }
    response = requests.get(NEWS_API_ENDPOINT, params=params)
    data = response.json()
    articles = data.get('articles', [])
    return articles

def get_football_news():
    params = {
        'apiKey': NEWS_API_KEY,
        'q': 'football',
        'category': 'sports',
        'pageSize': 10,
    }
    response = requests.get(NEWS_API_ENDPOINT, params=params)
    data = response.json()
    articles = data.get('articles', [])
    return articles




@app.route('/news')
def news():
    cricket_news = get_cricket_news()
    football_news = get_football_news()

    return render_template('news.html', cricket_news=cricket_news, football_news=football_news, )
#-------------------------------------
#Merch Page
# Sample product data
# Product data
products = [
    #Jersey
    {"id": 1, "name": "Cricket World Cup Jersey", "price": 1200.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTmWujjyiOtj__3mLcyvCfwd3I9-H9BN8B_UnzfraDeywGbDenKbOlpWQvfPF-4S-UNtCk&usqp=CAU" , "desc": "World Cup Edition jersey of the Indian Team for ages 8-10yrs"},
    {"id": 2, "name": "Cricket World Cup Jersey", "price": 1200.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTmWujjyiOtj__3mLcyvCfwd3I9-H9BN8B_UnzfraDeywGbDenKbOlpWQvfPF-4S-UNtCk&usqp=CAU" , "desc": "World Cup Edition jersey of the Indian Team for ages 11-14yrs"},
    {"id": 3, "name": "Cricket World Cup Jersey", "price": 1200.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTmWujjyiOtj__3mLcyvCfwd3I9-H9BN8B_UnzfraDeywGbDenKbOlpWQvfPF-4S-UNtCk&usqp=CAU" , "desc": "World Cup Edition jersey of the Indian Team for adults"},
    #Jersey 
    {"id": 4, "name": "Cricket Original Jersey", "price": 2000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQWcKY3c2Vi7Yuk--SPMf5vcQOr4hDY6oDK7lXHPgIraclbUjLwiuBTbQYj1rKnUYrzcMI&usqp=CAU" , "desc": "Original jersey of the Indian Team for ages 8-10yrs"},
    {"id": 5, "name": "Cricket Original Jersey", "price": 2000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQWcKY3c2Vi7Yuk--SPMf5vcQOr4hDY6oDK7lXHPgIraclbUjLwiuBTbQYj1rKnUYrzcMI&usqp=CAU" , "desc": "Original jersey of the Indian Team for ages 11-14yrs"},
    {"id": 6, "name": "Cricket Original Jersey", "price": 2000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQWcKY3c2Vi7Yuk--SPMf5vcQOr4hDY6oDK7lXHPgIraclbUjLwiuBTbQYj1rKnUYrzcMI&usqp=CAU" , "desc": "Original jersey of the Indian Team for adults"},
    
    #kit
    {"id": 7, "name": "Cricket Kit", "price": 8000.0, "image": "https://m.media-amazon.com/images/I/31ZFYLtoVeL._AC_UF894,1000_QL80_.jpg", "desc": "Contains gloves, glove inners, helmet, thigh pad, leg pad, supporter, abdominal guard. For ages 8-10"},
    {"id": 8, "name": "Cricket Kit", "price": 8000.0, "image": "https://m.media-amazon.com/images/I/31ZFYLtoVeL._AC_UF894,1000_QL80_.jpg", "desc": "Contains gloves, glove inners, helmet, thigh pad, leg pad, supporter, abdominal guar. For ages 11-14"},
    {"id": 9, "name": "Cricket Kit", "price": 8000.0, "image": "https://m.media-amazon.com/images/I/31ZFYLtoVeL._AC_UF894,1000_QL80_.jpg", "desc": "Contains gloves, glove inners, helmet, thigh pad, leg pad, supporter, abdominal guard. For adults"},
    
    {"id": 10, "name": "Leather Ball", "price": 4200.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSAMFVabrhawSOhz1Qtb35QbfH2AtcpBtNSPaSQbD45GQGOqyPDRlli8Ik06je4SbddswU&usqp=CAU" , "desc": "Set of 3 leather balls for cricket"},
    
    # shoes 
    {"id": 11, "name": "Cricket Spikes shoes by SG", "price": 3000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSIgNzoYEg6XPtyqHg4EoH5OY5sT4yC8K2OzHSmvqWhzR0T1r_nGf3DgesCi34Mg3lv5DI&usqp=CAU" , "desc": "Cricket shoes size 8"},
    {"id": 12, "name": "Cricket Spikes shoes by SG", "price": 3000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSIgNzoYEg6XPtyqHg4EoH5OY5sT4yC8K2OzHSmvqWhzR0T1r_nGf3DgesCi34Mg3lv5DI&usqp=CAU" , "desc": "Cricket shoes size 9"},
    {"id": 13, "name": "Cricket Spikes shoes by SG", "price": 3000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSIgNzoYEg6XPtyqHg4EoH5OY5sT4yC8K2OzHSmvqWhzR0T1r_nGf3DgesCi34Mg3lv5DI&usqp=CAU" , "desc": "Cricket shoes size 10"},
    {"id": 14, "name": "Cricket Spikes shoes by SG", "price": 3000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSIgNzoYEg6XPtyqHg4EoH5OY5sT4yC8K2OzHSmvqWhzR0T1r_nGf3DgesCi34Mg3lv5DI&usqp=CAU" , "desc": "Cricket shoes size 11"},
    {"id": 15, "name": "Cricket Spikes shoes by SG", "price": 3000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSIgNzoYEg6XPtyqHg4EoH5OY5sT4yC8K2OzHSmvqWhzR0T1r_nGf3DgesCi34Mg3lv5DI&usqp=CAU" , "desc": "Cricket shoes size 12"},
    {"id": 16, "name": "Cricket Spikes shoes by SG", "price": 3000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSIgNzoYEg6XPtyqHg4EoH5OY5sT4yC8K2OzHSmvqWhzR0T1r_nGf3DgesCi34Mg3lv5DI&usqp=CAU" , "desc": "Cricket shoes size 13"},
    {"id": 17, "name": "Cricket Spikes shoes by SG", "price": 3000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSIgNzoYEg6XPtyqHg4EoH5OY5sT4yC8K2OzHSmvqWhzR0T1r_nGf3DgesCi34Mg3lv5DI&usqp=CAU" , "desc": "Cricket shoes size 14"},

    
    #shoes   
    
    {"id": 18, "name": "Cricket Kookabura shoes", "price": 4500.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTelBCioyROakoTYN_OHbBg0VQ0Ce4AnEP8Den2mtaYsqTibv1-oEr6phZlrIJXgXdnko8&usqp=CAU" , "desc": "Cricket shoes size 8"},
    {"id": 19, "name": "Cricket Kookabura shoes", "price": 4500.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTelBCioyROakoTYN_OHbBg0VQ0Ce4AnEP8Den2mtaYsqTibv1-oEr6phZlrIJXgXdnko8&usqp=CAU" , "desc": "Cricket shoes size 9"},
    {"id": 19, "name": "Cricket Kookabura shoes", "price": 4500.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTelBCioyROakoTYN_OHbBg0VQ0Ce4AnEP8Den2mtaYsqTibv1-oEr6phZlrIJXgXdnko8&usqp=CAU" , "desc": "Cricket shoes size 10"},
    {"id": 20, "name": "Cricket Kookabura shoes", "price": 4500.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTelBCioyROakoTYN_OHbBg0VQ0Ce4AnEP8Den2mtaYsqTibv1-oEr6phZlrIJXgXdnko8&usqp=CAU" , "desc": "Cricket shoes size 11"},
    {"id": 21, "name": "PCricket Kookabura shoes", "price":4500.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTelBCioyROakoTYN_OHbBg0VQ0Ce4AnEP8Den2mtaYsqTibv1-oEr6phZlrIJXgXdnko8&usqp=CAU" , "desc": "Cricket shoes size 12"},
    {"id": 22, "name": "Cricket Kookabura shoes", "price": 4500.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTelBCioyROakoTYN_OHbBg0VQ0Ce4AnEP8Den2mtaYsqTibv1-oEr6phZlrIJXgXdnko8&usqp=CAU" , "desc": "Cricket shoes size 13"},
    {"id": 23, "name": "Cricket Kookabura shoes", "price": 4500.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTelBCioyROakoTYN_OHbBg0VQ0Ce4AnEP8Den2mtaYsqTibv1-oEr6phZlrIJXgXdnko8&usqp=CAU" , "desc": "Cricket shoes size 14"},

    #Bats
    
    {"id": 24, "name": "Cricket Bat Kashmir Willow", "price": 2500.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSYIspkDQlvvdIhBjwSWYLQipDYLDcanNZw9tV65KZO4dFHfuThx1WyV1BfEIiZCvkGytE&usqp=CAU" , "desc": "Cricket Bat Kashmir Willow by SG"},
    {"id": 25, "name": "Cricket Bat Kashmir Willow", "price": 3000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcToebuYSRn2gtakXeOLSQltx0yGxLrzo8kK1B2YL2QDmeBUFAwULGqoKQa23oNn7qgWdP0&usqp=CAU" , "desc": "Cricket Bat Kashmir Willow by SS"},
    {"id": 26, "name": "Cricket Bat Kashmir Willow", "price": 2800.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR8d1n_OeqFznGkyej7LOrKcvJ8isNbT5t9TEviTvjf1cT5Xn9XFgqlA5xbDq1I7aSCYSg&usqp=CAUx" , "desc": "Cricket Bat Kashmir Willow by Kookabura"},
    {"id": 27, "name": "Cricket Bat English Willow", "price": 2800.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSDPb-Dg2E-mCimrRSSJutxU6pEIgQpwcLPVFebykLMZpts74buBeK6aaY5tbcFraQ6lkQ&usqp=CAU" , "desc": "Cricket Bat English Willow by Kookabura"},
    {"id": 28, "name": "Cricket Bat English Willow", "price": 10000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ_RvRLD5CPCN_b0ahTlTsB-C5Ie4jvUvwpQCN1W0ldbbfx2YlZ52EfGwbvV9mEezDITjk&usqp=CAU" , "desc": "Cricket Bat English Willow by SS"},
    {"id": 29, "name": "Cricket Bat English Willow", "price": 6000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRKQfESsBs4DWtXmSdgMjj4RDmuP08fBBQpcj1en-q1_C1NppKwI5PgAfiGiU9LmGvKOWw&usqp=CAU" , "desc": "Cricket Bat English Willow by SG"},
    
    #hiking

    {"id": 30, "name": "Small Hiking Kit", "price": 12000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT-g6KeyoeKeUNHE_-5EjnKF6ICqp9lriHUSHULHm3c8o-CgKx0MwQEoYiGaKb-JcO5VXU&usqp=CAU"},
    {"id": 31, "name": "Big Hiking Kit", "price": 26000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQpMxBV9ZMsb09cceiy5XsNDf3JAulgY1ZjlBnfpZEm6pMnR5vUMPbM0whyVhaOyVBeVyQ&usqp=CAU"},
    
    #Football
    {"id": 32, "name": "Football Jersey", "price": 2000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRAAssEaEhpRKReZ__S2-Ins1d1FbhOuD6rGS5juC47edyZZs5jIibMv-V0TdCWANYzXHM&usqp=CAU" , "desc": "Football jersey for ages 8-10yrs"},
    {"id": 33, "name": "Football Jersey", "price": 2000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRAAssEaEhpRKReZ__S2-Ins1d1FbhOuD6rGS5juC47edyZZs5jIibMv-V0TdCWANYzXHM&usqp=CAU" , "desc": "Football jersey for ages 11-14yrs"},
    {"id": 34, "name": "Football Jersey", "price": 2000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRAAssEaEhpRKReZ__S2-Ins1d1FbhOuD6rGS5juC47edyZZs5jIibMv-V0TdCWANYzXHM&usqp=CAU" , "desc": "Football jersey for adults"},
    
    {"id": 35, "name": "Football Stockings", "price": 400.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSYPww4hbSY-cibSn6JpqW4oZ02gtJ2aa9NsVqk_VQmZhNvosVc-eNcRXRWVzWtR6ZMUjw&usqp=CAU"},
    {"id": 36, "name": "Football Stockings", "price": 400.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRq-S_cnrvBIRRVoNqLkb0d0H18pIkscZFVo_DVc0mXjUCnfk4pWCKH4cFa_d_D6pOgR8Q&usqp=CAU"},
    {"id": 37, "name": "Football Stockings", "price": 400.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRdCnLw8AdCFP1nxke3e2dDG0FNhLr6lnqB-PJ52i3c4C9MqCPRhmp69uRe5uTiyT4uwHQ&usqp=CAU"},

    
    #shoes
    {"id": 38, "name": "Football stud shoes ", "price": 2500.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSRz29fjif7O-_XbjYa530JPnXJ6KnlwxwRq1MkJJ1Sdshhkbiltr8CTEoY7nPEJ8gbAYw&usqp=CAU" , "desc": "Football stud shoes size 8"},
    {"id": 39, "name": "Football stud shoes ", "price": 2500.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSRz29fjif7O-_XbjYa530JPnXJ6KnlwxwRq1MkJJ1Sdshhkbiltr8CTEoY7nPEJ8gbAYw&usqp=CAU" , "desc": "Football stud shoes size 9"},
    {"id": 40, "name": "Football stud shoes ", "price": 2500.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSRz29fjif7O-_XbjYa530JPnXJ6KnlwxwRq1MkJJ1Sdshhkbiltr8CTEoY7nPEJ8gbAYw&usqp=CAU" , "desc": "Football stud shoes size 10"},
    {"id": 41, "name": "Football stud shoes ", "price": 2500.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSRz29fjif7O-_XbjYa530JPnXJ6KnlwxwRq1MkJJ1Sdshhkbiltr8CTEoY7nPEJ8gbAYw&usqp=CAU" , "desc": "Football stud shoes size 11"},
    {"id": 42, "name": "Football stud shoes ", "price": 2500.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSRz29fjif7O-_XbjYa530JPnXJ6KnlwxwRq1MkJJ1Sdshhkbiltr8CTEoY7nPEJ8gbAYw&usqp=CAU" , "desc": "Football stud shoes size 12"},
    {"id": 43, "name": "Football stud shoes ", "price": 2500.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSRz29fjif7O-_XbjYa530JPnXJ6KnlwxwRq1MkJJ1Sdshhkbiltr8CTEoY7nPEJ8gbAYw&usqp=CAU" , "desc": "Football stud shoes size 13"},
    {"id": 44, "name": "Football stud shoes ", "price": 2500.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSRz29fjif7O-_XbjYa530JPnXJ6KnlwxwRq1MkJJ1Sdshhkbiltr8CTEoY7nPEJ8gbAYw&usqp=CAU" , "desc": "Football stud shoes size 14"},
    
    {"id": 45, "name": "Football stud shoes ", "price": 2500.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRWhkxBKpS1G2m7TpeyqZnIMkd5MCMm33luvM8sDfTO42ZSFnEr96Ui4y1d0I4Xip_ddTI&usqp=CAU" , "desc": "Football stud shoes size 8"},
    {"id": 46, "name": "Football stud shoes ", "price": 2500.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRWhkxBKpS1G2m7TpeyqZnIMkd5MCMm33luvM8sDfTO42ZSFnEr96Ui4y1d0I4Xip_ddTI&usqp=CAU" , "desc": "Football stud shoes size 9"},
    {"id": 47, "name": "Football stud shoes ", "price": 2500.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRWhkxBKpS1G2m7TpeyqZnIMkd5MCMm33luvM8sDfTO42ZSFnEr96Ui4y1d0I4Xip_ddTI&usqp=CAU" , "desc": "Football stud shoes size 10"},
    {"id": 48, "name": "Football stud shoes ", "price": 2500.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRWhkxBKpS1G2m7TpeyqZnIMkd5MCMm33luvM8sDfTO42ZSFnEr96Ui4y1d0I4Xip_ddTI&usqp=CAU" , "desc": "Football stud shoes size 11"},
    {"id": 49, "name": "Football stud shoes ", "price": 2500.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRWhkxBKpS1G2m7TpeyqZnIMkd5MCMm33luvM8sDfTO42ZSFnEr96Ui4y1d0I4Xip_ddTI&usqp=CAU" , "desc": "Football stud shoes size 12"},
    {"id": 50, "name": "Football stud shoes ", "price": 2500.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRWhkxBKpS1G2m7TpeyqZnIMkd5MCMm33luvM8sDfTO42ZSFnEr96Ui4y1d0I4Xip_ddTI&usqp=CAU" , "desc": "Football stud shoes size 13"},
    {"id": 51, "name": "Football stud shoes ", "price": 2500.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRWhkxBKpS1G2m7TpeyqZnIMkd5MCMm33luvM8sDfTO42ZSFnEr96Ui4y1d0I4Xip_ddTI&usqp=CAU" , "desc": "Football stud shoes size 14"},


    {"id": 52, "name": "Shin Pads", "price": 800.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTaXoVu9hSGUFlzwhsR9HyVyh3W5uiZBNWvqreldRAJrjW7wv7FdRGFoRq-in-xrYome_g&usqp=CAU", "desc":" Shin Pads by Nike"},
    {"id": 53, "name": "Shin Pads", "price": 800.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR6EDaE0sC__INQOCI5nq0SsrP8PLV0EysLpugaC1M0jD-CNRNp25icr-1Yrh8rcmDlwWM&usqp=CAU", "desc":"Shin Pads by Adidas"},
    
    {"id": 54, "name": "Goal keeper Gloves", "price": 2000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQZQ5jRbtZInj3jc0UjLLl5q5UNerhJofcZsrHuMvhGw5Lmxtpj9Mgql7BYEwlx-paJhrI&usqp=CAU", "desc":" Goalkeeper Gloves by Nivia"},
    {"id": 55, "name": "Goal keeper Gloves", "price": 2000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTR9qZJdNYE9pOGs4sN5J_5oV5LJbacBP-plHIWmU4FOoMVbS2LdCm9ZIer7Tm-1Hl9JFQ&usqp=CAU", "desc":" Goalkeeper Gloves by Vector X"},
    
    {"id": 56, "name": "Football", "price": 5000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRPXXY7YrhlkdLm9ydOom0D3bH6Jy8UaXvNU6TGdaAEVGpKNGotLfw1_8VgpJ_NP6KyrAw&usqp=CAU", "desc":" Football by Kipsta"},
    {"id": 57, "name": "Football", "price": 5000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS-n3-vIXVPDSMliHbEdS9GjhEwjZvaUTF-UhTtCh9NC65os4xhOvEprx_LOZVUPE7_nw4&usqp=CAU", "desc":" Football by Nivia"},
    
    
    #Basketball 
    {"id": 58, "name": "Basketball Jersey", "price": 2500.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSvH834xqBhN5Si0Kl3yH_txc8dbGfHc-JsvGTDFlV6Bo14C3Z4mIrFyK19fJZsDXR7PVs&usqp=CAU" , "desc": "Basketball jersey  for ages 8-10yrs"},
    {"id": 59, "name": "Basketball Jersey", "price": 2500.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSvH834xqBhN5Si0Kl3yH_txc8dbGfHc-JsvGTDFlV6Bo14C3Z4mIrFyK19fJZsDXR7PVs&usqp=CAU" , "desc": "Basketball jersey  for ages 11-14yrs"},
    {"id": 60, "name": "Basketball Jersey", "price": 2500.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSvH834xqBhN5Si0Kl3yH_txc8dbGfHc-JsvGTDFlV6Bo14C3Z4mIrFyK19fJZsDXR7PVs&usqp=CAU" , "desc": "Basketball jersey  for adults"},

    {"id": 61, "name": "Basketball Jersey", "price": 2500.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRi4NFE2TySJGz7IvDP4n8SyiXb1js3Hbq-ko0_5mzzYB5JpEUhoMYoa6D3zuUdJS7OA5I&usqp=CAU" , "desc": "Basketball jersey  for ages 8-10yrs"},
    {"id": 62, "name": "Basketball Jersey", "price": 2500.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRi4NFE2TySJGz7IvDP4n8SyiXb1js3Hbq-ko0_5mzzYB5JpEUhoMYoa6D3zuUdJS7OA5I&usqp=CAU" , "desc": "Basketball jersey  for ages 11-14yrs"},
    {"id": 63, "name": "Basketball Jersey", "price": 2500.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRi4NFE2TySJGz7IvDP4n8SyiXb1js3Hbq-ko0_5mzzYB5JpEUhoMYoa6D3zuUdJS7OA5I&usqp=CAU" , "desc": "Basketball jersey  for adults"},
    
    {"id": 64, "name": "Basketball shoes ", "price": 10000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSmAweBqwLNnd7QB0F_G7KIk5sEk7aYVAgXlD49PGSEWiTgH88MdoVy6YXdBzk28OFvHmc&usqp=CAU" , "desc": "Basketball shoes Black-White color scheme size 8"},
    {"id": 65, "name": "Basketball shoes ", "price": 10000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSmAweBqwLNnd7QB0F_G7KIk5sEk7aYVAgXlD49PGSEWiTgH88MdoVy6YXdBzk28OFvHmc&usqp=CAU" , "desc": "Basketball shoes Black-White color scheme size 9"},
    {"id": 66, "name": "Basketball shoes ", "price": 10000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSmAweBqwLNnd7QB0F_G7KIk5sEk7aYVAgXlD49PGSEWiTgH88MdoVy6YXdBzk28OFvHmc&usqp=CAU" , "desc": "Basketball shoes Black-White color scheme size 10"},
    {"id": 67, "name": "Basketball shoes ", "price": 10000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSmAweBqwLNnd7QB0F_G7KIk5sEk7aYVAgXlD49PGSEWiTgH88MdoVy6YXdBzk28OFvHmc&usqp=CAU" , "desc": "Basketball shoes Black-White color scheme size 11"},
    {"id": 68, "name": "Basketball shoes ", "price": 10000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSmAweBqwLNnd7QB0F_G7KIk5sEk7aYVAgXlD49PGSEWiTgH88MdoVy6YXdBzk28OFvHmc&usqp=CAU" , "desc": "Basketball shoes Black-White color scheme size 12"},
    {"id": 69, "name": "Basketball shoes ", "price": 10000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSmAweBqwLNnd7QB0F_G7KIk5sEk7aYVAgXlD49PGSEWiTgH88MdoVy6YXdBzk28OFvHmc&usqp=CAU" , "desc": "Basketball shoes Black-White color scheme size 13"},
    {"id": 70, "name": "Basketball shoes ", "price": 10000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSmAweBqwLNnd7QB0F_G7KIk5sEk7aYVAgXlD49PGSEWiTgH88MdoVy6YXdBzk28OFvHmc&usqp=CAU" , "desc": "Basketball shoes Black-White color scheme size 14"},

    {"id": 71, "name": "Basketball shoes ", "price": 12000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSYub8kwSxJv2X9V3l8gqH3g5Wh1FLMzBRUJIEF5FVrKq9tB5qBgnnnzZGPdLGge6OHJrg&usqp=CAU" , "desc": "Basketball shoes Black-Red color scheme size 8"},
    {"id": 72, "name": "Basketball shoes ", "price": 12000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSYub8kwSxJv2X9V3l8gqH3g5Wh1FLMzBRUJIEF5FVrKq9tB5qBgnnnzZGPdLGge6OHJrg&usqp=CAU" , "desc": "Basketball shoes Black-Red color scheme size 9"},
    {"id": 73, "name": "Basketball shoes ", "price": 12000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSYub8kwSxJv2X9V3l8gqH3g5Wh1FLMzBRUJIEF5FVrKq9tB5qBgnnnzZGPdLGge6OHJrg&usqp=CAU" , "desc": "Basketball shoes Black-Red color scheme size 10"},
    {"id": 74, "name": "Basketball shoes ", "price": 12000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSYub8kwSxJv2X9V3l8gqH3g5Wh1FLMzBRUJIEF5FVrKq9tB5qBgnnnzZGPdLGge6OHJrg&usqp=CAU" , "desc": "Basketball shoes Black-Red color scheme size 11"},
    {"id": 75, "name": "Basketball shoes ", "price": 12000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSYub8kwSxJv2X9V3l8gqH3g5Wh1FLMzBRUJIEF5FVrKq9tB5qBgnnnzZGPdLGge6OHJrg&usqp=CAU" , "desc": "Basketball shoes Black-Red color scheme size 12"},
    {"id": 76, "name": "Basketball shoes ", "price": 12000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSYub8kwSxJv2X9V3l8gqH3g5Wh1FLMzBRUJIEF5FVrKq9tB5qBgnnnzZGPdLGge6OHJrg&usqp=CAU" , "desc": "Basketball shoes Black-Red color scheme size 13"},
    {"id": 77, "name": "Basketball shoes ", "price": 12000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSYub8kwSxJv2X9V3l8gqH3g5Wh1FLMzBRUJIEF5FVrKq9tB5qBgnnnzZGPdLGge6OHJrg&usqp=CAU" , "desc": "Basketball shoes Black-Red color scheme size 14"},
    
    {"id": 78, "name": "Basketball", "price": 2000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSLY_Tecu2Gcjq6sjZOIBOPKRX3A2FlHDaKf3Pyz9pWL1jfrBIgC6zclsfdu1oU_Ku5PqY&usqp=CAU", "desc":"Spalding Basketball"},
    #Kabaddi
    {"id": 79, "name": "Kabaddi Jersey", "price": 1500.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRAy5-b--p2uL5aDvkcnqD9sGEflryfL_eKVKkzIuyNCAR1G2fd_ICTYEEUup1M16l8ZMk&usqp=CAU" , "desc": "Kabaddi jersey  for ages 8-10yrs"},
    {"id": 80, "name": "Kabaddi Jersey", "price": 1500.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRAy5-b--p2uL5aDvkcnqD9sGEflryfL_eKVKkzIuyNCAR1G2fd_ICTYEEUup1M16l8ZMk&usqp=CAU" , "desc": "Kabaddi jersey  for ages 11-14yrs"},
    {"id": 81, "name": "Kabaddi Jersey", "price": 1500.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRAy5-b--p2uL5aDvkcnqD9sGEflryfL_eKVKkzIuyNCAR1G2fd_ICTYEEUup1M16l8ZMk&usqp=CAU" , "desc": "Kabaddi jersey  for adults"},

    {"id": 82, "name": "Kabaddi Jersey", "price": 1500.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSAZaKQLIjpEQDl9PfHSIRHRGu5_QpXRsKS4Wqt_hJ-_wMVevLatnFj2_nJstHytkknkjs&usqp=CAU" , "desc": "Kabaddi jersey  for ages 8-10yrs"},
    {"id": 83, "name": "Kabaddi Jersey", "price": 1500.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSAZaKQLIjpEQDl9PfHSIRHRGu5_QpXRsKS4Wqt_hJ-_wMVevLatnFj2_nJstHytkknkjs&usqp=CAU" , "desc": "Kabaddi jersey  for ages 11-14yrs"},
    {"id": 84, "name": "Kabaddi Jersey", "price": 1500.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSAZaKQLIjpEQDl9PfHSIRHRGu5_QpXRsKS4Wqt_hJ-_wMVevLatnFj2_nJstHytkknkjs&usqp=CAU" , "desc": "Kabaddi jersey  for adults"},
    
    {"id": 85, "name": "Kabaddi shoes ", "price": 1000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSTX5pypoQ_yr3oxiGFbTW4qJfN7KO2WJ1AgVhcUCCY1wE5p8w2KDoygl3Q_7bR6DrCig4&usqp=CAU" , "desc": "Kabaddi shoes Black-White color scheme size 8"},
    {"id": 86, "name": "Kabaddi shoes ", "price": 1000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSTX5pypoQ_yr3oxiGFbTW4qJfN7KO2WJ1AgVhcUCCY1wE5p8w2KDoygl3Q_7bR6DrCig4&usqp=CAU" , "desc": "Kabaddi shoes Black-White color scheme size 9"},
    {"id": 87, "name": "Kabaddi shoes ", "price": 1000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSTX5pypoQ_yr3oxiGFbTW4qJfN7KO2WJ1AgVhcUCCY1wE5p8w2KDoygl3Q_7bR6DrCig4&usqp=CAU" , "desc": "Kabaddi shoes Black-White color scheme size 10"},
    {"id": 88, "name": "Kabaddi shoes ", "price": 1000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSTX5pypoQ_yr3oxiGFbTW4qJfN7KO2WJ1AgVhcUCCY1wE5p8w2KDoygl3Q_7bR6DrCig4&usqp=CAU" , "desc": "Kabaddi shoes Black-White color scheme size 11"},
    {"id": 89, "name": "Kabaddi shoes ", "price": 1000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSTX5pypoQ_yr3oxiGFbTW4qJfN7KO2WJ1AgVhcUCCY1wE5p8w2KDoygl3Q_7bR6DrCig4&usqp=CAU" , "desc": "Kabaddi shoes Black-White color scheme size 12"},
    {"id": 90, "name": "Kabaddi shoes ", "price": 1000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSTX5pypoQ_yr3oxiGFbTW4qJfN7KO2WJ1AgVhcUCCY1wE5p8w2KDoygl3Q_7bR6DrCig4&usqp=CAU" , "desc": "Kabaddi shoes Black-White color scheme size 13"},
    {"id": 91, "name": "Kabaddi shoes ", "price": 1000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSTX5pypoQ_yr3oxiGFbTW4qJfN7KO2WJ1AgVhcUCCY1wE5p8w2KDoygl3Q_7bR6DrCig4&usqp=CAU" , "desc": "Kabaddi shoes Black-White color scheme size 14"},
    
    {"id": 92, "name": "Kabaddi shoes ", "price": 1000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTO3N9DiSxfyNiqWrk8jUuwajRw_XOQnyZcmGGUixrONYPBc5_zz5EGDv5vQFhMh_aXMO0&usqp=CAU" , "desc": "Kabaddi shoes Red-White color scheme size 8"},
    {"id": 93, "name": "Kabaddi shoes ", "price": 1000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTO3N9DiSxfyNiqWrk8jUuwajRw_XOQnyZcmGGUixrONYPBc5_zz5EGDv5vQFhMh_aXMO0&usqp=CAU" , "desc": "Kabaddi shoes Red-White color scheme size 9"},
    {"id": 94, "name": "Kabaddi shoes ", "price": 1000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTO3N9DiSxfyNiqWrk8jUuwajRw_XOQnyZcmGGUixrONYPBc5_zz5EGDv5vQFhMh_aXMO0&usqp=CAUU" , "desc": "Kabaddi shoes Red-White color scheme size 10"},
    {"id": 95, "name": "Kabaddi shoes ", "price": 1000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTO3N9DiSxfyNiqWrk8jUuwajRw_XOQnyZcmGGUixrONYPBc5_zz5EGDv5vQFhMh_aXMO0&usqp=CAU" , "desc": "Kabaddi shoes Red-White color scheme size 11"},
    {"id": 96, "name": "Kabaddi shoes ", "price": 1000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTO3N9DiSxfyNiqWrk8jUuwajRw_XOQnyZcmGGUixrONYPBc5_zz5EGDv5vQFhMh_aXMO0&usqp=CAU" , "desc": "Kabaddi shoes Red-White color scheme size 12"},
    {"id": 97, "name": "Kabaddi shoes ", "price": 1000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTO3N9DiSxfyNiqWrk8jUuwajRw_XOQnyZcmGGUixrONYPBc5_zz5EGDv5vQFhMh_aXMO0&usqp=CAU" , "desc": "Kabaddi shoes Red-White color scheme size 13"},
    {"id": 98, "name": "Kabaddi shoes ", "price": 1000.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTO3N9DiSxfyNiqWrk8jUuwajRw_XOQnyZcmGGUixrONYPBc5_zz5EGDv5vQFhMh_aXMO0&usqp=CAU" , "desc": "Kabaddi shoes Red-White color scheme size 14"},




]

# Cart data
cart = {}


@app.route('/merch')
def merch():
    return render_template('merch.html', products=products, cart=cart)




@app.route('/add_to_cart', methods=['POST'])
def add_to_cart():
    product_id = int(request.form['product_id'])
    quantity = int(request.form['quantity'])

    cart[product_id] = cart.get(product_id, 0) + quantity

    return render_template('merch.html', products=products, cart=cart, total=get_cart_total())


@app.route('/remove_from_cart', methods=['POST'])
def remove_from_cart():
    product_id = int(request.form['product_id'])

    if product_id in cart:
        cart[product_id] -= 1

        if cart[product_id] <= 0:
            cart.pop(product_id, None)

    return render_template('merch.html', products=products, cart=cart, total=get_cart_total())


@app.route('/checkout', methods=['POST'])
def checkout():
    # Process the order (you can add your logic here)

    # Clear the cart after processing the order
    cart.clear()

    return render_template('merch.html', products=products, cart=cart, total=0, message="Order placed successfully!")


def get_cart_total():
    return sum(products[product_id - 1]['price'] * quantity for product_id, quantity in cart.items())


#-------------------------------------
# Route for the home page

@app.route('/stop_video_pushup', methods=['GET'])
def stop_video_pushup():
    global video_access_event_pushup
    global counterp
    counterp =0
    video_access_event_pushup.clear()
    return jsonify({"message": "Video access stopped."})

@app.route('/stop_video_squat', methods=['GET'])
def stop_video_squat():
    global video_access_event_squat
    global counters
    counters =0
    video_access_event_squat.clear()
    return jsonify({"message": "Video access stopped."})


# @app.route('/start_video_pushup', methods=['GET'])
# def start_video_pushup():
#     global video_access_event
#     video_access_event.set()
#     return jsonify({"message": "Video access started."})


@app.route('/')
def home():

    
    global y
    y = 0
    return render_template('home.html')

@app.route('/chat_bot')
def cht():
    global y
    y = 0
    return render_template('chtbot.html')


@app.route('/get_response', methods=['POST'])
def get_response():
    try:
        user_message = request.form['user_message']
        bot_response = generate_response(user_message)
        return jsonify({'bot_response': bot_response})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'bot_response': 'An error occurred. Please try again.'}), 500

def generate_response(user_message):
    # Simple rule-based responses for fitness chatbot
    if 'exercise' in user_message.lower():
        return "Exercise is crucial for maintaining a healthy lifestyle. What type of exercise are you interested in?"
    elif 'diet' in user_message.lower():
        return "A balanced diet is essential. What specific dietary information are you looking for?"
    elif 'motivation' in user_message.lower():
        return "Staying motivated is key! What are your fitness goals?"
    else:
        return "I'm sorry, I didn't understand that. Can you please provide more details?"

@app.route('/sportstraining')
def st():  
    return render_template('st.html')

@app.route('/fitness')
def fit():  
    return render_template('fit.html')


@app.route('/scores')
def scores():  
    return render_template('sc.html')

@app.route("/football")
def fb():
    return render_template('football.html')

@app.route("/cricket")
def cri():
    return render_template('cricket.html')

@app.route("/basketball")
def bb():
    return render_template('basketball.html')

@app.route("/kabaddi")
def kb():
    return render_template('kabaddi.html')

@app.route("/hiking")
def hi():
    return render_template('hiking.html')
      
   

@app.route('/puc')
def push():
    global video_access_event_pushup
    video_access_event_pushup.set()
    global y
    y = 1
    return render_template('pushup.html')
    

@app.route('/sqc')
def squat():
    global video_access_event_squat
    video_access_event_squat.set()
    global y
    y = 2
    return render_template('squat.html')
    

# Route for the video feed
@app.route('/video_feedp')
def video_feedp():
    return Response(stream_with_context(process_videop()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for the video feed
@app.route('/video_feeds')
def video_feeds():
    return Response(stream_with_context(process_videos()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

