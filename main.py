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
                    
            if  right <= 87:
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
    #Trophies
    
    

    
    
    #Jersey
    {"id": 1, "name": "Cricket World Cup Jersey", "price": 1200.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTmWujjyiOtj__3mLcyvCfwd3I9-H9BN8B_UnzfraDeywGbDenKbOlpWQvfPF-4S-UNtCk&usqp=CAU" , "desc": "World Cup Edition jersey of the Indian Team for ages 8-10yrs"},
    {"id": 1, "name": "Cricket World Cup Jersey", "price": 1200.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTmWujjyiOtj__3mLcyvCfwd3I9-H9BN8B_UnzfraDeywGbDenKbOlpWQvfPF-4S-UNtCk&usqp=CAU" , "desc": "World Cup Edition jersey of the Indian Team for ages 11-14yrs"},
    {"id": 1, "name": "Cricket World Cup Jersey", "price": 1200.0, "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTmWujjyiOtj__3mLcyvCfwd3I9-H9BN8B_UnzfraDeywGbDenKbOlpWQvfPF-4S-UNtCk&usqp=CAU" , "desc": "World Cup Edition jersey of the Indian Team for adults"},
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
    
    {"id": 99, "name": "Man of the Match Trophy ", "price": 1000.0, "image": "data:image/jpeg;base64,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G7B2QOcMw6Ejuakj1C9jvftkd3cJd5z56ysJM/72c1WooAsT313cGYz3VxKZiGl3yM3mEdC2TyR71HNNLOwaeWSRgAoLsWIA6DntUdFABX0N8J/2gbXwv4f07Qta0iVrW1Ty1ubZwWxnqVP17GvnmigD9FvBnjnw94ytfO8P6lDcsoy8OdsifVTzXT1+aWgazf6BqsGo6TcyW13CwZXRsZ9j6j2r9DvAevJ4n8IaTrEeP9Lt1kYDs2OR+eaAN+iiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAPmf9srXHS00LQ4yQsjPdSehx8q/zNfLVfpD4p8LaL4qsTaa9p8F5D/DvX5l91bqPwrxHxT+zFpF2Xk8OatcWDnkRXC+ag/Hg/rQB8mUV7NrX7OXjiwLGzWw1FB0MM+0/k2K4nU/hj410wn7X4b1JVH8SQlx+YzQBx9FaFzomq2mftWmX0OOvmQOv8xVJopF+9G6/VSKAGUUEgdSBSZHqPzoAWikyPUfnTlRm+6pb6DNACUVpWGgavqBAsdLvrgnp5UDN/IV3Xh34H+O9bZSujmyiP8Ay0vHEYx646n8qAPM6+x/2RdUubv4fXVjcRSiKyuiIZWUhWVhkgHvg5/OsrwN+zRpti8dz4t1BtQlBz9mtwUi+hPU/pX0BpenWelWENnp1tFbWsS7UiiXaqj6UAW6KKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACjFFFAEckEcn+sjVvqM1Tn0bTbj/AF9hayf70Sn+laFFAGBL4O8OSnMmh6a31tk/wqBvAXhRiS3h3Sif+vZP8K6aigDmk8C+Fo/ueHtLH0tU/wAKv2vhzRrUg22lWMRHTZAo/pWtRQBHHBHEMRxog/2RipAKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigD//Z" , "desc": "Man of the Match Trophy for every sport"},
    {"id": 100, "name": "Cricket Trophy", "price": 2000.0, "image": "data:image/jpeg;base64,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KCkqNDU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6g4SFhoeIiYqSk5SVlpeYmZqio6Slpqeoqaqys7S1tre4ubrCw8TFxsfIycrS09TV1tfY2drh4uPk5ebn6Onq8fLz9PX29/j5+v/EAB8BAAMBAQEBAQEBAQEAAAAAAAABAgMEBQYHCAkKC//EALURAAIBAgQEAwQHBQQEAAECdwABAgMRBAUhMQYSQVEHYXETIjKBCBRCkaGxwQkjM1LwFWJy0QoWJDThJfEXGBkaJicoKSo1Njc4OTpDREVGR0hJSlNUVVZXWFlaY2RlZmdoaWpzdHV2d3h5eoKDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uLj5OXm5+jp6vLz9PX29/j5+v/aAAwDAQACEQMRAD8A+qaKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKZK6Rxs8jBVUZLMcAD1rwrxF8UbvxLr7aJ4TuGstNCyCTUlUGSUqpOIs8Acfe/KplJR3NaVGdZ8sFc9xmuoIP9dNHH/vMBRDdQTH9zNHJ/usDXx5410q0SXxA1zc3k9xbopgM8ryliR82Sc49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rllsemaLqaalBvUbXU4ZfStGuD8E3My6o8MiKqSIeQ2eRXeV+hZVi5YrDqc99mfNYuiqNVxWwUUUV6RzBRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAVn3WkWdzO0s0O526nJrQorOpShVVqiuvMqM5Qd4uxlf2Bp3/PuPzNH9g6d/wA+4/M1q0Vj9Rw3/Ptfci/b1f5n95lf2Dp3/PuPzNOGh6f/AM+yfma06KPqOH/59r7kHt6j+0/vM3+xNP8A+fVP1o/sTT/+fVP1rSop/U8P/IvuQvbVP5n95Rt9Ls7eZZYYFSRehFXhRRWtOlCmrQVkTKTlrJ3CiiitCQooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigD//2Q==" , "desc": "Cricket Trophy  "},
    {"id": 101, "name": "Cricket Trophy", "price": 2000.0, "image": "data:image/jpeg;base64,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KCkqNDU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6g4SFhoeIiYqSk5SVlpeYmZqio6Slpqeoqaqys7S1tre4ubrCw8TFxsfIycrS09TV1tfY2drh4uPk5ebn6Onq8fLz9PX29/j5+v/EAB8BAAMBAQEBAQEBAQEAAAAAAAABAgMEBQYHCAkKC//EALURAAIBAgQEAwQHBQQEAAECdwABAgMRBAUhMQYSQVEHYXETIjKBCBRCkaGxwQkjM1LwFWJy0QoWJDThJfEXGBkaJicoKSo1Njc4OTpDREVGR0hJSlNUVVZXWFlaY2RlZmdoaWpzdHV2d3h5eoKDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uLj5OXm5+jp6vLz9PX29/j5+v/aAAwDAQACEQMRAD8A+qaKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACjNB6Vna3rFjoli95qdwsEC8bm7n0A7mgDRorB8L+K9J8TRSPpNz5pjxvRlKsuemQa3hRa24BRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUhpciszXtYs9D06S91CZYoU456sewA7mluAa9rFnoenSXmozLFCvc9WPYAdzXzX4z8YXfirVVmlkMVqjp5MCyMAo55OByfel8Z+L7vxXqiTSyeVaq6eRAjuAoyeTgdff8q7/wCE/wAPWC2+s66jdEe3t2cnGAcMwP14Fd1OMaC55b+qJeuho/BLwpe6VDLq2oF42uoESOEsSQo5yR2PoK9WFNQYGMYp1ck5ub5mNKwUUUVIwooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKM0UAFITzQTis7XdXs9E0+S91CURwpx6kk9AB3NG+gCa7rFlomnSXuoTCOFOPdiegA7mvmzxn4uvPFWqJNK/lWqOvkwKzgKMnk4HJ46/lR408XXnirVEmlbyrVWXyYF8wBRuIycdT7/lXf/Cf4ekJbazrsZzhXt7dmbggkhmB+vArtpxjQXPLcl66CfCb4esEt9Z11GzhXt7cs3GM4ZgfrwK9mUY7cUBcGnVyTqSqO8ikrBRRRUgFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABQaQmmSyrFG7ysFRRliTgAUmA/PNITXmmv8Ajm6N8U0gqsCHG5lyX/PtXOX3xQ8Q6fMBNa2M0JxtZY3B5OMHBPPesqNaFafs4vU1nRnCPM0eva7q9nounyXuoSeXAntkk9gB3NfNfjTxbeeKtUWabMdqjL5MAEmFG/GTjqeOv5Uzxf4q1HxXfRzXoCQoB5cKLJtX58Z9yfX+tehfCr4chVg1nXoRvwGgtzuGPmJDMCfyFenTjGgueTu9Opzu70E+FHw8+W31nXYucBre3JbghiQzA/XgV7MFxQFGMU7Fck6kqjvIpKwUUUVIwooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKQmo5pVhjaSRgqKMlieAKVwCaVIo2kkYKijJY8ACvLfF3iWTWJjaWTFbJTy3TzP8A61N8Z+KX1WQ2lkzLZqcEjrIf8K5tz5ESqvDHqa8nF4vm9yGx6WGw3L709yUCOBecf1NZF8Eu98BQOrHLZ6Cnyys77Izl+7elPjQRrhfz9a4FJxfMnqdjSaszn9K046b4p0xnRZLfz48NtY9ZOh5xnmvqdQNor5/tIHubuCOOPzZPMVlXHcHIr3+Hd5SbvvYGfrXv0MZLFQXPujycRRVKWnUfiiiitznCiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAozSGo5pUhiaSVgqKMsT0AoAJ5UijaSVlVFGSxPAFeU+NPFT6tK1nZMUslPLDrIf8Kb408UyatK1pZMUslOCe8h/wrlRgDAryMXi+b3IbHo4bD8vvy3AcVHPM8r7EOW7t/dpsjlm8uL73c/3aeirEmB+fcmvOO4ERYkwP/rk1NawS3M6xQqzyOcKq0ltBLdXCQwoXlc4VQM1634N8MR6PAJpwr3jjk/3PYV0YehKtKy2Ma1ZUl5h4P8ADMejwCacLJeuOW/uewrqV6UijAp1e7TpxprlieRKbm7sKKKKskKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigApN4pa8Q+Mfju7gv5ND0iZoRGP8ASJUOGJP8IPbj+dRUqKnG7OzA4Kpj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" , "desc": "Cricket Trophy"},
    
    
    {"id": 102, "name": "Basketball Trophy", "price": 1200.0, "image": "data:image/jpeg;base64,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9OjK2ltH0PMhRioWnNX1/E8x8UKdLS/sQD9mvGWeDjhW3Dcv9al+yPcXOtXFtzeWl0k0fqcKMj8a9Kk0W5bHmQK2OmcGohpEsTOywKrOcsQMbj71sqWNULewlf032/REPD0nO/Orflv+rPL5dQS4tfEU8bfLctCoJ7ZHP5YNb+hTMbiKIa1a3sSrgRBAH4HGK6oaUse4C2iG7qAo5qAaZDDKJVtIklGRvVAD+dZ1qeIs06Ml8r9EuxVKik0+dP8A4e/c82RbseH5pBZ2v2UTMDc4zKnzckfStnxYwlTSLeOUyo2DvJ5boAa6oabapYyWXkgWz53JnrnrXG6yiDxJa2sC7YrdVVVHYda6cC3XxcVKLVm3qj1siwrpV+a90kbCjC4HagnA55ooYrt5Gfxr9KhG7sdWYYieHoOcVdnY/DE7ru/J/uL/ADr0MdK88+GLE3d8OANi8D616GOldNrHxMpubcpbsKKKKBBRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFACN0rjL0SCW4EYHmbjjPHNdoearT2UE5zJGCfXoaYJ2dzzNfD1xI5aWdASck4Jq3H4YjP37lz9Frt20iL+B2X680w6Ww+7Ip+oqOSJ6Es1xT0UrGDo+iW2myvJBuLsMEnFbiU8WEq91NSLayjsPzqlGK2RwVatStLnqO7FSph0pixSD+GpVRsdKoysVJqoz960pYpD0U1TltZmziM0hmVL1NUJ62ZNPuj0hP5iqr6Rev0h/NhSGYU3esq5s4HlMpQeZ/exzXWN4ev3/AIYx9Wpv/CJ3j/elhX8zUuEXujalWqUXzU5WOKkh29DUDgAEGu/TwTuP76849FT/ABq9beC9MjIMokmI/vNgfkKORHW80xEoOE3dPuYXwvB+1X7YO3Yoz+NeiDpVe0s4LOIRWsSRIOyjFWB0qjzwooooGFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAYooooAOKKKKACjFFFABijFFFABijFFFABiiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAP/Z" , "desc": "Basketball Trophy"},
    {"id": 103, "name": "Basketball Trophy", "price": 1200.0, "image": "data:image/jpeg;base64,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" , "desc": "Basketball Trophy"},
    
    
    {"id": 104, "name": "Football Trophy", "price": 1200.0, "image": "data:image/jpeg;base64,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KCkqNDU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6g4SFhoeIiYqSk5SVlpeYmZqio6Slpqeoqaqys7S1tre4ubrCw8TFxsfIycrS09TV1tfY2drh4uPk5ebn6Onq8fLz9PX29/j5+v/EAB8BAAMBAQEBAQEBAQEAAAAAAAABAgMEBQYHCAkKC//EALURAAIBAgQEAwQHBQQEAAECdwABAgMRBAUhMQYSQVEHYXETIjKBCBRCkaGxwQkjM1LwFWJy0QoWJDThJfEXGBkaJicoKSo1Njc4OTpDREVGR0hJSlNUVVZXWFlaY2RlZmdoaWpzdHV2d3h5eoKDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uLj5OXm5+jp6vLz9PX29/j5+v/aAAwDAQACEQMRAD8A+qaKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKDQAUVm2+t6bPq0+lw39s+owANLbLIPMQHoSvWtKgAooooAKKKKAE3UbhXzL+1t8R9T0Sey8M6FdyWjzw+fdyxNtcqSQqA9hwSa+b/DXj7xP4d1KO+0vWb1JUYEq0pZHHowPBFAH6Vg5orkvhb4s/wCE18DaZrpgMElyh8yPsHUkNj2yOK62gAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKRqWigD8+PiLq+u+GPjXq+p+bc2mow3zSRuSQTHnjr1Ur+FfdHgjxJY+K/Ddlq2nXEc0c8Ss+xgdj4+ZT6EGqHj/4f+HvHGnPba7YRyS7cR3KjEsR9Vb+nSvkbxFpPjX9n/xULjSbyR9KuG/dS4zDcD+7IvQN/kUAfc9FfOHgj9qHR754bfxVp8mnSNgNcQnzIs+pHUD869lT4h+EWgWUeJNJEZQOCbpB8p79aAOqozXmPiH45eA9G097ka3BfODtEFmfMdj9Ow968m1T9q9A+NM8Nkru6z3HUfQCgDI/bS8OvBr+keIYxmG6i+yyezrkj8wf0r5qFej/ABg+LGrfEq5tlvIIbOwtSWit4iT8x6sxPU4rg9JuIrTU7W4ubcXMMUqu8JbaHAOSue2aAP0X+DumvpPwy8N2cjFnSyjJJXb1GcY/Guxr5v8ADX7Unh+SGOLV9GvLHbhQbciVAv6GvefC/iPS/FOjwapod3HdWcw+V17HuCOxHpQBr0UUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABWT4m0DTfEuj3Gl6zaR3VlOMOjj9Qex961qKAPjzx/wDsxaxaXjzeDrmO+s2ORBO+yVPbPRv0rzHxV8HvGvhfSpdS1bR3Wyi/1kkbq+wepwc4r9Eagv7SC+s5rW7iWa3mQxyRsMhlIwQaAPyzPB4or7l1f9m7wRdaXNBZw3VpdMSyXCzFipPQYPBFfHPj3wxd+DfFmoaFflXmtJNu9Rw6kZVh9QRQBz9FFFAC16J8H/ijqvw41nzbYm40ucj7TZs3DD+8vow9a4PT7Sa/vbeztl3z3EixRrnGWY4A/M1Z13RtQ0DVJtO1e0ltbyE7XjkXB+o9R70Afo54C8a6N430WPUtCullQgCSInEkTejL2rpq/Pj4SeMtN+HetW2sTi8vrmRfmgtrjy40U8HeP42746DivvTw3rNp4g0Oy1bTnL2l3EJYyeuD2PvQBp0UUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAEZr5A/bP8AC7W3iLS/EcKHyruL7NMwH8acr+YJ/Kvr+vOP2gPC3/CV/DHVrWJN11bp9qg453Jzj8RkUAfnlRSng0lAG14KXf4x0JP71/AP/Ii19zfG74UWPxD0JpYEjg1+3j/0a5PG7HOxvY/pXwf4fuksdd067kzsguY5Wx1wrAn+Vfp3p9zFe2Ntc27b4Zo1kRh3VgCD+RoA+Pfhh+zhrd7rxfxxb/YtJhDApHMDJM2MDaRnA75NfWnhXQLHwxoFno+kxtHZWqbIwzZP1J9a1sUooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKjmjEsbxuMqwII+tSUUAfmb8Q9HOgeOdd0sqVW2u5EUf7O4lf0Irna+i/2wfBM1h4qh8UWkTGz1BBHOyjhJVGOfTIA/EGvnUjFAAOor9MPhzIZfAPhxz1OnW//AKLWvzPFfod+z4zv8IPDTPcyXBNv95xyPmI2/QdKAPRaKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigDO1/R7LXtHu9M1OFZrS5jMcikdiOo96/N74heHW8KeM9X0Ri7CznZEZhgsnVT+RFfpg7KFJJAHvXyD+2la6eNf0K9thF9smhdJmQjLBSNufzNAHzXX3n+y/4rsNf+Gljp9sFiu9KX7PPED7kh/oc/nmvg4KSCQDgda96/Y2u5YviddW6uwinsJN654JUqR/M0AfbFFAooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAoNFI3SgDkfFvxI8L+EdVtdO8QapHaXVynmIrKSAucZJA4rnPGHx18EeHbAzR6pHqVwVzHb2R3lvqeg/GvM/wBrb4bX+psfGenSiSK0t1iurc9VUH76+vXkV87/AAw8HXHjnxnYaJAZEjlYmeZF3eVGOrf0/GgDrPiZ8c/FPjSaSG3mbSdK6LbWzkEj/bfqf5V51FYaxrCtPDa316qdXVHkA/Hmvuzwx8CfAWhQRKdGTUJ1HM16xkLH12/d/SvSLHTbKwt1gsrSC3hUYCRIFUfgKAPhTw58LfEX/CsPFOt3umTWyJFG8CzJtd1V8uQDzgCsn9n/AMUp4T+KGk3c5Atp2NpMT2V+M/gcGv0GubeK4tpLeZFeGRSjoRwykYIr88fjN4Jn8A+PbuwQN9kdvtFnJ6xk5A+oPH4UAfomhBGQcilrjPhB4mi8WfDzRtUSRXmaBY5wD92RRhgfxH612dABRQTxxXmPxA+NfhHwRfNYajdTXV+n37e0QOyf7xJAB9s0AenUV5t8PfjL4T8d3Ys9Kupbe/IyttdIEdh/s8kH869JFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUjdKAOP+IHxA0fwRHb/wBq+fJNcAmKKFMkgdeeg615XqX7RiqSNN0BmHZrifb+gBry/wAX6/qd58QdQm1i8e5it7mSNEboiBiAFHQVtRa14aukAuY7Qt/00gAP54oA1rr9obxI5P2fTdMiHuHY/wAxWXP8ePGch+SWxiHotuD/ADzTZLLwrcj9zDZFz0CykZ/DNYN5p2ihtqW6Lk4BErdfzoEWNc+Lfi/WdNurC91CJrW5jaKRBboMqRg9q4PwLf6t4M1C4vNHvzbzyp5W9FGduc9/oK9I0nwjpF8QJYLpD/0zl/xBrWl+G+hsv7t9RjPqXU/+y0Aco/xU8at/zMF4PoQP6VE3xO8aH/mY78f8DrqV+GukA/NdX5HsV/wpk/w70RASLjUx+KH/ANlpgcx/ws3xp/0Mmof9/K5Px/ruseKLGOTWtQnvZLXLRmY5Kg9Rmux1Xw3pFjnDai59yg/pXJa3p8VxZSxWPmI7DAMrgj36CgDC8EfEHxF4NWWHRNTntraZg0kaNwSO+PWvSLL4zeMZUDQ+IJX9mCkj8MV5WnhS7P3poR+JNbWi+FI4JRJdi4nccqITsH45BzSGejzfGvx1DayldWViEOMwIe30rwO+up727luruRpZ5nLu7HJZick162dLgaPaul3hY9S0mQfyUVxer+CdSSd5LO2PkE5CswBUenNAHOaTf3Gl6jbX1lK0VzbyLJG6nBDA5FfW2i/tITeRCNU0BT8o3PBPyeOuCP618wWvh6W2mV9SKIgOdisGY/lWvJf2qHCo5NAH2z8PvipoXja9aysVube+CGTyZl6gdcEcV6COlfnl4d8Q6lpms293o08lncKwAdDg4z0PqPav0E053lsLaSQ5do1Zj7kUAWaKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKDRRQB81/Ev4Na2+t32qaFsvoLmRpTDnbIhJyRzwa8d1nw1rWlMy6jpV7bkdS8Rx+fSvvbFNdFdSrqGU9QRmgD86LhWXggg+/FVi8gAAkcAcgZr9BdQ8I+H9RJN7ounzE9S0C5/PFc/d/CLwPdEl9AtUJ/55ll/kaAPia21K+hbdHdzq3qHNW11zU1bcNQug3r5hr6/b4HeAyf+QQw+k7/AONN/wCFG+BP+gTJ/wCBMn+NAHyK/iLVz11G6P8A20NM/wCEm1lDlNSuQR/tmvr3/hRngP8A6BMn/gTJ/jSH4FeAz/zCZP8AwJk/xoA+PbrxZrsylZNQkYe4B/pWJNcXE0heSRix6mvtv/hQ/gE9dIk/8CZP8aT/AIUP4A/6BEn/AIEyf40AfEqXFwhBSRgR3FS/2jf/APP3N/32a+1f+FDeAP8AoESf+BMn+NH/AAobwB/0CJP/AAJk/wAaAPic3t43W6nI/wB81CzysctI5PuTX3APgR4BB/5A7n63En+NXbT4NeBLYgp4ft3x/wA9GZv5mgD4SCMx/iJrX0nw1rGqyBNO0u8uWP8AzzhYj88V97af4I8MacQbPQNNiI6EW6k/mRW/FDHCgSJFRB0CgAUAfIvw/wDgR4lv723uNajTS7NXDMJCGkIBzgKOn419dW8YihSNeiKFH4VJiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooA//Z" , "desc": "World Cup Edition jersey of the Indian Team for ages 8-10yrs"},
    
    {"id": 105, "name": "Football Trophy", "price": 1200.0, "image": "data:image/jpeg;base64,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iiigAooooAKKKKACiiigAooooAKKKKACuY8f+CdH8c6I+m65bCROsco4eJv7yntXT0UAfAXxV+CniPwNPLcRQPqWjZJW7gUnYP8AbXqv16V5SQR1r9UXRXUqwBUjBB6EV5p4v+CXgfxM7y3OkJaXLnJlsz5RJ9cDj9KAPz4pcV9g3f7KegPKzW2v6lEnZWRGx+OKvaP+y54UtZQ+oajqV8Ac7Cyxg/kM0AfHFjY3WoXUdtY28txcSHakcSFmY+wFfTHwZ/Zynllg1fx8nlwjDx6aD8zenmHsPavovwh4D8NeEYwugaRbWsmMGULukP1Y810+KAIrS2itLeOC2jSKGNQqIgwFA6ACpqKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACvnf9r/AFXU9D0vw7f6NqN3Y3AnkQtBKUyNoPOOvSvoivmP9tu6VdG8N2ufmaaWTHsFA/rQB4fa/G34hWybV8S3Tj/poqMfzIqnqfxc8d6lGyXPibUNjdVjcR/+ggVwlFAH6YfDnf8A8IH4f813kk+wwlnc5LHYMkk966OuX+F9wt38O/Dc6nIewhP/AI4K6igAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAD0r42/bS1YXHjTSNMRgfsloXYehdv8AAV9kMQFJJwB1NfnF8Z/EX/CUfEvXdSR90BuDFCe3lp8o/PGfxoA4mlHWkooA+/8A9mPVRqnwd0UZy9oHtWz/ALLcfoRXqtfKn7FfiMf8Tzw9K/Py3cIP/fLf0r6rHSgAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACig0maAFooyPWjI9aAPOvj14tXwj8NdVu0kCXdwhtrf13vxkfQZNfnk5LMSeSeSa/RT4n/AAz0n4jLZR63eX0UNoWKR27hVLHucg84rgf+GXfBP/P7q/8A3+T/AOJoA+J6K+2P+GXfBP8Az+6v/wB/k/8A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" , "desc": "World Cup Edition jersey of the Indian Team for ages 8-10yrs"},
    {"id": 106, "name": "Football Trophy", "price": 1200.0, "image": "" , "desc": "World Cup Edition jersey of the Indian Team for ages 8-10yrs"},



]

# Cart data
cart = {}


@app.route('/merch')
def merch():
    
    
    return render_template('merch.html', products=products, cart=cart)

# @app.route('/searchmerch')
# def search():


#     search_query = request.args.get('search', '').lower()

#     # Filter products based on the search query
#     filtered_products = [product for product in products if search_query in product['name'].lower()]

#     return render_template('merch.html', products=filtered_products, cart=cart, total=get_cart_total())



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
        return "A balanced diet is essential.Make sure to eat more protein and less fat intake whilst working out regularly"
    elif 'motivation' in user_message.lower():
        return "Staying motivated is key! What are your fitness goals?"
    elif 'cardio' in user_message.lower():
        return "Jogging, Skipping , Jumping, Mountain climbers are good examples of cardio workouts"    
    elif 'biceps' in user_message.lower() or "bicep" in user_message.lower():
        return "Bicep curls, Hammer curls, pull ups are some good exercises for growing biceps"   
    elif 'workout' and "age" in user_message.lower() or "working out" and "age" in user_message.lower():
        return "There is no strict age to start working out. But if you are below 14 it is advised that you do not start going to the gym yet"   
    elif 'okay' in user_message.lower() or "ok" in user_message.lower():
        return "Do you have any other queries?" 
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

if __name__ == '__main__':
    app.run(debug=True)
