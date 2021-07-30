from tabula import read_pdf
from pdf2image import convert_from_path
from tabulate import tabulate
import tensorflow as tf
import luminoth as lm
import numpy as np
import cv2 as cv
import utils
from table import Table
from PIL import Image
import sys
import img2pdf as imp


image_list = []

if len(sys.argv) < 2:
    print("give proper arguments ")
    sys.exit(1)

paths = sys.argv[1] 
if paths.endswith(".png") or paths.endswith(".jpg"):
	im = Image.open(paths)
	rgb_im = im.convert('RGB')
	rgb_im.save('colors.jpg')


else :
	  img = Image.open(paths)
	  new_img = img.resize( (1366, 780) )
	  new_img.save('colors.jpg')	
        

path = 'colors.jpg'

image = cv.imread(path)


NUM_CHANNELS = 3
if len(image.shape) == NUM_CHANNELS:
    grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)


MAX_THRESHOLD_VALUE = 255
BLOCK_SIZE = 15
THRESHOLD_CONSTANT = -3

filtered = cv.adaptiveThreshold(~grayscale, MAX_THRESHOLD_VALUE, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, BLOCK_SIZE, THRESHOLD_CONSTANT)


SCALE = 30


horizontal = filtered.copy()
vertical = filtered.copy()

horizontal_size = int(horizontal.shape[1] / SCALE)
horizontal_structure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
utils.isolate_lines(horizontal, horizontal_structure)

vertical_size = int(vertical.shape[0] / SCALE)
vertical_structure = cv.getStructuringElement(cv.MORPH_RECT, (1, vertical_size))
utils.isolate_lines(vertical, vertical_structure)


mask = horizontal + vertical
(contours, _) = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)


intersections = cv.bitwise_and(horizontal, vertical)




tables = [] # list of tables
for i in range(len(contours)):
   
    (rect, table_joints) = utils.verify_table(contours[i], intersections)
    if rect == None or table_joints == None:
        continue

   
    table = Table(rect[0], rect[1], rect[2], rect[3])

    
    joint_coords = []
    for i in range(len(table_joints)):
        joint_coords.append(table_joints[i][0][0])
    joint_coords = np.asarray(joint_coords)

    
    sorted_indices = np.lexsort((joint_coords[:, 0], joint_coords[:, 1]))
    joint_coords = joint_coords[sorted_indices]

 
    table.set_joints(joint_coords)

    tables.append(table)

    
sss=input("Enter Output file name : ")
out = "extracted/"

table_name = sss
psm = 6
oem = 3
mult = 3

count = 1

for table in tables[::-1]:
	
    table.print_joints()
    table_entries = table.get_table_entries()
    table_roi = image[table.y:table.y + table.h, table.x:table.x + table.w]
    table_roi = cv.resize(table_roi, (table.w * mult, table.h * mult))
    
    
    countname = str(count)
    final_name = table_name+" - "+countname+".jpg"
    cv.imwrite(out + final_name, table_roi)
    image_list.append(out+final_name)
    

    count = count+1

print("Total Number of Tables Found = "+str(count-1))


for imagePath in image_list:
    cv.namedWindow("output", cv.WINDOW_NORMAL) 
    image = cv.imread(imagePath)
    ims = cv.resize(image, (1368, 600))
    cv.imshow("output", ims)
    
    key = cv.waitKey(0)
    
    

