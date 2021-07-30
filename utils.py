import cv2 as cv
import pytesseract as tess
from PIL import Image
import subprocess as s
import os


def isolate_lines(src, structuring_element):
	cv.erode(src, structuring_element, src, (-1, -1))
	cv.dilate(src, structuring_element, src, (-1, -1))


MIN_TABLE_AREA = 50
EPSILON = 5 

def verify_table(contour, intersections):
    area = cv.contourArea(contour)

    if (area < MIN_TABLE_AREA):
        return (None, None)

    curve = cv.approxPolyDP(contour, EPSILON, True)

    
    rect = cv.boundingRect(curve)

    possible_table_region = intersections[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
    (possible_table_joints, _) = cv.findContours(possible_table_region, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    if len(possible_table_joints) < 10:
        return (None, None)

    return rect, possible_table_joints




