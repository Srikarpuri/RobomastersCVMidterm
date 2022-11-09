#!/usr/bin/env python
# coding: utf-8

# In[23]:


import cv2
import torch
import os
import numpy as np


def det_move(x_coord, y_coord, xres, yres):
    y_coord = yres-y_coord
    centerx, centery = xres/2.0, yres/2.0

    x = x_coord-centerx
    y = y_coord-centery
    if(x != 0):
        x /= centerx
    if(y != 0):
        y /= centery
    return x,y


def is_blue(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, (0, 4, 226),(60, 255, 255))

    # Get Blue masks
    blue_mask = cv2.inRange(hsv,(68, 38, 131), (113, 255, 255))
        # Get the contours of the red and blue regions
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Determine which is greater
    if len(blue_contours) > 0:
        redArea = 0
        blueArea = 0
        for c in red_contours:
            redArea += cv2.contourArea(c)
        for c in blue_contours:
            blueArea += cv2.contourArea(c)
        if blueArea > redArea:
            #print("blue detected")
            return True
    return False    

class plate_detector:

    # Constructor
    def __init__(self, repo='ultralytics/yolov5', model='custom', model_file_path='best.pt'):
        

    
        self.model = torch.hub.load(repo, 'custom', path=model_file_path)


    def process_frame(self, color_image, conf_thres=0.25, k=1, debug=False):
      
        results = self.model(color_image)

        detections_rows = results.pandas().xyxy

        for i in range(len(detections_rows)):
            rows = detections_rows[i].to_numpy()

        top_k_detections = rows[:k]

        bboxes = []


        for i in range(len(top_k_detections)):
            x_min, y_min, x_max, y_max, conf, cls, label = top_k_detections[i]


            bboxes.append([x_min, y_min, x_max, y_max, conf, label])

        return bboxes
     #mask = object_detector.apply(frame)
    #_,mask = cv2.threshold(mask,254,255,cv2.THRESH_BINARY)
    #contours,_=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #for cnt in contours:
     #   area = cv2.contourArea(cnt)
      #  if area > 500:
       #     #cv2.drawContours(frame,[cnt],-1,(0,255,0),2)
        #    x,y,w,h = cv2.boundingRect(cnt)
         #   cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)


# In[25]:



cap = cv2.VideoCapture("vid.mp4")

#object_detector = cv2.createBackgroundSubtractorMOG2(history = 100, varThreshold = 10)

#flag =Flagger("RED",(0, 4, 226),(60, 255, 255),(68, 38, 131),(113, 255, 255))
pd = plate_detector()
while True:    
    ret, frame = cap.read()
    height,width,_ = frame.shape
    if is_blue(frame):
        bboxes= pd.process_frame(frame)
        for box in bboxes:
            x,y,x2,y2,conf,label = box
            center_x = (x+x2)/2
            center_y = (y+y2)/2
            offset_x,offset_y = det_move(center_x,center_y, 360,640)
            cv2.rectangle(frame,(int(x),int(y)),(int(x2),int(y2)),(0,255,0),3)
            cv2.putText(frame,str(round(conf,5)),(int(x),int(y)-15),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),2)
            angle_offset = "Horizontal Offset: " + str(round(offset_x,5)) + "Vertical Offset: " + str(round(offset_y,5))             
            cv2.putText(frame,angle_offset,(int(width)-630,int(height)-10),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),2)
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:




