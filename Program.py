<<<<<<< HEAD
import cv2 #OpenCv
import numpy as np #numpy 
import pre_processing as pp # preprocessing module
import segmentation as seg  # segmentation module
import detection as det #detection module
import ROI # get Region Of Interest
from glob import glob


#kennel size
getkennel = det.getKennel()

#Shape Detector
class ShapeDetector:
	def __init__(self):
		pass

	def detect(self, c):
		# initialize the shape name and approximate the contour
		shape = "unidentified"
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.05 *peri, False)
        		# if the shape is a triangle, it will have 3 vertices
		if len(approx) == 3:
			shape = "triangle"
		# if the shape has 4 vertices, it is either a square or
		# a rectangle
		elif len(approx) == 4:
			# compute the bounding box of the contour and use the
			# bounding box to compute the aspect ratio
			(x, y, w, h) = cv2.boundingRect(approx)
			ar = w / float(h)
			# a square will have an aspect ratio that is approximately
			# equal to one, otherwise, the shape is a rectangle
			shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
		# if the shape is a pentagon, it will have 5 vertices
		elif len(approx) == 5:
			shape = "pentagon"
		# otherwise, we assume the shape is a circle
		else:
			shape = "circle"

		# return the name of the shape
		return shape

#initiate shape classifier
shape_detector = ShapeDetector()




###################################################################################################################
#otsu detection	
def Otsu_Detection(mypath):



    #read image
    for fn in glob(mypath + "\\*.jpg"):
        image = cv2.imread(fn)

        #grayscale image
        gray = pp.grayImage(image)

        #filter image
        Filter = pp.Gfilter(gray)

        #segment image

        segment = seg.otsu(Filter)

        #detect edges
        edged = seg.CannySeg(segment)

        #dilate segmented image
        gradient = cv2.morphologyEx(edged, cv2.MORPH_GRADIENT, det.getKennel())

        #get contours of the image
        contours = det.getContours(gradient ,image)


        #calculate contours area
        getArea = det.get_contour_areas(contours)

        # Labeling Contours left to right
        for (i,c) in enumerate(contours):

            hull = cv2.convexHull(c) 

            M = cv2.moments(hull)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            #if shape looks like circle/pentagon, treat as pothole
            
            shape_detector_start = ShapeDetector.detect(shape_detector,hull)
            if (shape_detector_start == "circle" or shape_detector_start == "pentagon" or shape_detector_start == "rectangle") and getArea[i] > 3250:
                draw = cv2.drawContours(image, [hull], 0, (0, 0, 255), 2)
                cv2.putText(image, "Pothole", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)          
            else:
                continue

            cv2.imshow("result",image)
            cv2.waitKey(0)

        cv2.destroyAllWindows()

###################################################################################################################
#Global Thresholding
def Global_Threshold(mypath):

    

    #read image
    for fn in glob(mypath + "\\*.jpg"):
        image = cv2.imread(fn)

        #grayscale image
        gray = pp.grayImage(image)

        #filter image
        Filter = pp.Gfilter(gray)

        #segment image
        segment = seg.global_threshold(Filter)
 

        #detect edges
        edged = seg.CannySeg(segment)

        #dilate segmented image
        gradient = cv2.morphologyEx(edged, cv2.MORPH_GRADIENT, det.getKennel())

        #get contours of the image
        contours = det.getContours(gradient ,image)
        
        getArea = det.get_contour_areas(contours)

        # Labeling Contours left to right
        for (i,c) in enumerate(contours):

            hull = cv2.convexHull(c) 

            M = cv2.moments(hull)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            #if shape looks like circle/pentagon, treat as pothole
            
            shape_detector_start = ShapeDetector.detect(shape_detector,hull)
            if (shape_detector_start == "circle" or shape_detector_start == "pentagon" or shape_detector_start == "rectangle") and getArea[i] > 3250:
                draw = cv2.drawContours(image, [hull], 0, (0, 0, 255), 2)
                cv2.putText(image, "Pothole", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)          
            else:
                continue

            cv2.imshow("result",image)
            cv2.waitKey(0)

        cv2.destroyAllWindows()

###################################################################################################################
        #Canny Detection
def Canny_detection(mypath):


    #read image
    for fn in glob(mypath + "\\*.jpg"):
        image = cv2.imread(fn)

        #grayscale image
        gray = pp.grayImage(image)

        #filter image
        Filter = pp.Gfilter(gray)

        #detect edges
        edged = seg.CannySeg(Filter)

        #dilate segmented image
        gradient = cv2.morphologyEx(edged, cv2.MORPH_GRADIENT, det.getKennel())

        #get contours of the image
        contours = det.getContours(gradient ,image)
        
        getArea = det.get_contour_areas(contours)

        # Labeling Contours left to right
        for (i,c) in enumerate(contours):

            hull = cv2.convexHull(c) 

            M = cv2.moments(hull)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            #if shape looks like circle/pentagon, treat as pothole
            
            shape_detector_start = ShapeDetector.detect(shape_detector,hull)
            if (shape_detector_start == "circle" or shape_detector_start == "pentagon" or shape_detector_start == "rectangle") and getArea[i] > 3250:
                draw = cv2.drawContours(image, [hull], 0, (0, 0, 255), 2)
                cv2.putText(image, "Pothole", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)          
            else:
                continue

            cv2.imshow("result",image)
            cv2.waitKey(0)

        cv2.destroyAllWindows()

###################################################################################################################
#K_means clustering
def Kmeans_detection(mypath):

    #read image
    for fn in glob(mypath + "\\*.jpg"):
        image = cv2.imread(fn)

        #grayscale image
        gray = pp.grayImage(image)

        #filter image
        Filter = pp.Gfilter(gray)

        #segment image

        segment = seg.K_means(Filter,3,10)
        cv2.imshow("segmentation", segment)

                #detect edges
        edged = seg.CannySeg(segment)
        cv2.imshow("EDGED" , edged)
        #dilate segmented image
        gradient = cv2.morphologyEx(edged, cv2.MORPH_GRADIENT, det.getKennel())

        #get contours of the image
        contours = det.getContours(gradient ,image)
        
        getArea = det.get_contour_areas(contours)

        # Labeling Contours left to right
        for (i,c) in enumerate(contours):

            hull = cv2.convexHull(c) 

            M = cv2.moments(hull)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            #if shape looks like circle/pentagon, treat as pothole
            
            shape_detector_start = ShapeDetector.detect(shape_detector,hull)
            if (shape_detector_start == "circle" or shape_detector_start == "pentagon" or shape_detector_start == "rectangle") and getArea[i] > 3250:
                draw = cv2.drawContours(image, [hull], 0, (0, 0, 255), 2)
                cv2.putText(image, "Pothole", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)          
            else:
                continue

            cv2.imshow("result",image)
            cv2.waitKey(0)

        cv2.destroyAllWindows()


###################################################################################################################
#haar-cascade classifier
def haarClasifier_image(mypath, classifier_path):
    #create a classifier
    pothole_classifier = cv2.CascadeClassifier(classifier_path)
    for fn in glob(mypath + "\\*.jpg"):
        # read image
        image = cv2.imread(fn)

        #create detect multi scale
        pothole = pothole_classifier.detectMultiScale(image,50,5)

        #filtering image
        for (x,y,w,h) in pothole:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image,'pothole',(x,y), font, 0.5, (0,0,255), 2, cv2.LINE_AA)
            
        cv2.imshow('Result',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


=======
import cv2 #OpenCv
import numpy as np #numpy 
import pre_processing as pp # preprocessing module
import segmentation as seg  # segmentation module
import detection as det #detection module
import ROI # get Region Of Interest
from glob import glob


#kennel size
getkennel = det.getKennel()

#Shape Detector
class ShapeDetector:
	def __init__(self):
		pass

	def detect(self, c):
		# initialize the shape name and approximate the contour
		shape = "unidentified"
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.05 *peri, False)
        		# if the shape is a triangle, it will have 3 vertices
		if len(approx) == 3:
			shape = "triangle"
		# if the shape has 4 vertices, it is either a square or
		# a rectangle
		elif len(approx) == 4:
			# compute the bounding box of the contour and use the
			# bounding box to compute the aspect ratio
			(x, y, w, h) = cv2.boundingRect(approx)
			ar = w / float(h)
			# a square will have an aspect ratio that is approximately
			# equal to one, otherwise, the shape is a rectangle
			shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
		# if the shape is a pentagon, it will have 5 vertices
		elif len(approx) == 5:
			shape = "pentagon"
		# otherwise, we assume the shape is a circle
		else:
			shape = "circle"

		# return the name of the shape
		return shape

#initiate shape classifier
shape_detector = ShapeDetector()




###################################################################################################################
#otsu detection	
def Otsu_Detection(mypath):



    #read image
    for fn in glob(mypath + "\\*.jpg"):
        image = cv2.imread(fn)

        #grayscale image
        gray = pp.grayImage(image)

        #filter image
        Filter = pp.Gfilter(gray)

        #segment image

        segment = seg.otsu(Filter)

        #detect edges
        edged = seg.CannySeg(segment)

        #dilate segmented image
        gradient = cv2.morphologyEx(edged, cv2.MORPH_GRADIENT, det.getKennel())

        #get contours of the image
        contours = det.getContours(gradient ,image)


        #calculate contours area
        getArea = det.get_contour_areas(contours)

        # Labeling Contours left to right
        for (i,c) in enumerate(contours):

            hull = cv2.convexHull(c) 

            M = cv2.moments(hull)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            #if shape looks like circle/pentagon, treat as pothole
            
            shape_detector_start = ShapeDetector.detect(shape_detector,hull)
            if (shape_detector_start == "circle" or shape_detector_start == "pentagon" or shape_detector_start == "rectangle") and getArea[i] > 3250:
                draw = cv2.drawContours(image, [hull], 0, (0, 0, 255), 2)
                cv2.putText(image, "Pothole", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)          
            else:
                continue

            cv2.imshow("result",image)
            cv2.waitKey(0)

        cv2.destroyAllWindows()

###################################################################################################################
#Global Thresholding
def Global_Threshold(mypath):

    

    #read image
    for fn in glob(mypath + "\\*.jpg"):
        image = cv2.imread(fn)

        #grayscale image
        gray = pp.grayImage(image)

        #filter image
        Filter = pp.Gfilter(gray)

        #segment image
        segment = seg.global_threshold(Filter)
 

        #detect edges
        edged = seg.CannySeg(segment)

        #dilate segmented image
        gradient = cv2.morphologyEx(edged, cv2.MORPH_GRADIENT, det.getKennel())

        #get contours of the image
        contours = det.getContours(gradient ,image)
        
        getArea = det.get_contour_areas(contours)

        # Labeling Contours left to right
        for (i,c) in enumerate(contours):

            hull = cv2.convexHull(c) 

            M = cv2.moments(hull)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            #if shape looks like circle/pentagon, treat as pothole
            
            shape_detector_start = ShapeDetector.detect(shape_detector,hull)
            if (shape_detector_start == "circle" or shape_detector_start == "pentagon" or shape_detector_start == "rectangle") and getArea[i] > 3250:
                draw = cv2.drawContours(image, [hull], 0, (0, 0, 255), 2)
                cv2.putText(image, "Pothole", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)          
            else:
                continue

            cv2.imshow("result",image)
            cv2.waitKey(0)

        cv2.destroyAllWindows()

###################################################################################################################
        #Canny Detection
def Canny_detection(mypath):


    #read image
    for fn in glob(mypath + "\\*.jpg"):
        image = cv2.imread(fn)

        #grayscale image
        gray = pp.grayImage(image)

        #filter image
        Filter = pp.Gfilter(gray)

        #detect edges
        edged = seg.CannySeg(Filter)

        #dilate segmented image
        gradient = cv2.morphologyEx(edged, cv2.MORPH_GRADIENT, det.getKennel())

        #get contours of the image
        contours = det.getContours(gradient ,image)
        
        getArea = det.get_contour_areas(contours)

        # Labeling Contours left to right
        for (i,c) in enumerate(contours):

            hull = cv2.convexHull(c) 

            M = cv2.moments(hull)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            #if shape looks like circle/pentagon, treat as pothole
            
            shape_detector_start = ShapeDetector.detect(shape_detector,hull)
            if (shape_detector_start == "circle" or shape_detector_start == "pentagon" or shape_detector_start == "rectangle") and getArea[i] > 3250:
                draw = cv2.drawContours(image, [hull], 0, (0, 0, 255), 2)
                cv2.putText(image, "Pothole", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)          
            else:
                continue

            cv2.imshow("result",image)
            cv2.waitKey(0)

        cv2.destroyAllWindows()

###################################################################################################################
#K_means clustering
def Kmeans_detection(mypath):

    #read image
    for fn in glob(mypath + "\\*.jpg"):
        image = cv2.imread(fn)

        #grayscale image
        gray = pp.grayImage(image)

        #filter image
        Filter = pp.Gfilter(gray)

        #segment image

        segment = seg.K_means(Filter,3,10)
        cv2.imshow("segmentation", segment)

                #detect edges
        edged = seg.CannySeg(segment)
        cv2.imshow("EDGED" , edged)
        #dilate segmented image
        gradient = cv2.morphologyEx(edged, cv2.MORPH_GRADIENT, det.getKennel())

        #get contours of the image
        contours = det.getContours(gradient ,image)
        
        getArea = det.get_contour_areas(contours)

        # Labeling Contours left to right
        for (i,c) in enumerate(contours):

            hull = cv2.convexHull(c) 

            M = cv2.moments(hull)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            #if shape looks like circle/pentagon, treat as pothole
            
            shape_detector_start = ShapeDetector.detect(shape_detector,hull)
            if (shape_detector_start == "circle" or shape_detector_start == "pentagon" or shape_detector_start == "rectangle") and getArea[i] > 3250:
                draw = cv2.drawContours(image, [hull], 0, (0, 0, 255), 2)
                cv2.putText(image, "Pothole", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)          
            else:
                continue

            cv2.imshow("result",image)
            cv2.waitKey(0)

        cv2.destroyAllWindows()


###################################################################################################################
#haar-cascade classifier
def haarClasifier_image(mypath, classifier_path):
    #create a classifier
    pothole_classifier = cv2.CascadeClassifier(classifier_path)
    for fn in glob(mypath + "\\*.jpg"):
        # read image
        image = cv2.imread(fn)

        #create detect multi scale
        pothole = pothole_classifier.detectMultiScale(image,50,5)

        #filtering image
        for (x,y,w,h) in pothole:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image,'pothole',(x,y), font, 0.5, (0,0,255), 2, cv2.LINE_AA)
            
        cv2.imshow('Result',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


>>>>>>> ceda3c94e9f899b85a037487778c8a9092903e6e
