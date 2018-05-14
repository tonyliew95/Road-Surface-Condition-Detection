import cv2
import numpy as np



def getKennel():
    kernel = np.ones((5,5),np.uint8)
    return (kernel)
    
def get_contour_areas(contours):
    # returns the areas of all contours as list
    all_areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        all_areas.append(area)
    return (all_areas)

def getContours(segImage,oriImage):
    _,contours, hierarchy = cv2.findContours(segImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return (contours)

def drawContours(contours,oriImage):
    draw = cv2.drawContours(oriImage, contours, -1, (0,0,255), 2)
    return (draw)

def label_contour_center(image, c):
    # Places a red circle on the centers of contours
    M = cv2.moments(c)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
 
    # Draw the countour number on the image
    cv2.circle(image,(cx,cy), 5, (0,0,255), -1)
    return image


def x_cord_contour(contours):
    #Returns the X cordinate for the contour centroid
    if cv2.contourArea(contours) > 10:
        M = cv2.moments(contours)
        return (int(M['m10']/M['m00']))



