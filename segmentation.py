<<<<<<< HEAD
import cv2
import numpy as np
import pre_processing

#method 1
def CannySeg(frame):
    edged = cv2.Canny(frame,100,150)
    return (edged)

#method 2
def K_means(frame,cluster,interation):
    # image reshape 
    Z = frame.reshape((-1,1))
    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, interation, 1.0)
    K = cluster
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((frame.shape))
    return (res2)
   
#method 3
def global_threshold(frame):
    #intensity = 255
    #threshold = 127
    ret,thresh = cv2.threshold(frame,127,255,cv2.THRESH_BINARY)
    return (thresh)

#method 4
def otsu(frame):
    #intensity = 255
    #threshold = 0
    ret,thresh = cv2.threshold(frame,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return (thresh)



=======
import cv2
import numpy as np
import pre_processing

#method 1
def CannySeg(frame):
    edged = cv2.Canny(frame,100,150)
    return (edged)

#method 2
def K_means(frame,cluster,interation):
    # image reshape 
    Z = frame.reshape((-1,1))
    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, interation, 1.0)
    K = cluster
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((frame.shape))
    return (res2)
   
#method 3
def global_threshold(frame):
    #intensity = 255
    #threshold = 127
    ret,thresh = cv2.threshold(frame,127,255,cv2.THRESH_BINARY)
    return (thresh)

#method 4
def otsu(frame):
    #intensity = 255
    #threshold = 0
    ret,thresh = cv2.threshold(frame,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return (thresh)



>>>>>>> ceda3c94e9f899b85a037487778c8a9092903e6e
