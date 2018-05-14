import cv2
import numpy as np

def crop(frame):
    #get height and with of frame
    height, width = frame.shape[:2]

    #get Region of Interest
    start_row, start_col = int(height * .25), int(width * .25)

    # Let's get the ending pixel coordinates (bottom right)
    end_row, end_col = int(height * .75), int(width * .75)

    # Simply use indexing to crop out the rectangle we desire
    cropped = frame[start_row:end_row , start_col:end_col]
    #croping the image

    cv2.rectangle(frame, (start_col,start_row), (end_col,end_row), (255,0,0), 3)
    
    return (cropped)



