import cv2
import numpy as np

#input is image/frame

def resize(frame):
    #resize input into 0.5 
    resize_frame = cv2.resize(frame, None , fx = 0.5, fy = 0.5)
    return(resize_frame)

def grayImage(resize_frame):
    #grayscale input
    gray = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2GRAY)
    return (gray)

def Gfilter(gray_frame):
    stdX = np.std(gray_frame)
    #filtering using Gaussian Filtering
    Gfiltering = cv2.GaussianBlur(gray_frame,(5,5),0)
    return(Gfiltering)
    
