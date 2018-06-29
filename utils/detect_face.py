#CREDIT: https://github.com/JeeveshN/Face-Detect; MLP group project: https://github.com/AdeelMufti/TensorNinjas/blob/master/detect_face.py
from random import randint
import cv2
import sys
import os

MIN_FACE_SIZE = 32
PADDING = 32
CASCADE="haarcascade_frontalface_alt_tree.xml"
FACE_CASCADE=cv2.CascadeClassifier(CASCADE)

def detect_faces(filename, image_path):
	if (os.path.isfile(filename)):
		return
	try:
		image=cv2.imread(image_path)
		image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		faces = FACE_CASCADE.detectMultiScale(image_grey,scaleFactor=1.16,minNeighbors=1,minSize=(MIN_FACE_SIZE,MIN_FACE_SIZE),flags=0)
		num_faces = len(faces)
		height, width, channels = image.shape

		for i,(x,y,w,h) in enumerate(faces):
			y0 = (y-PADDING) if y-PADDING>=0 else 0
			y1 = (y+h+PADDING) if y+h+PADDING <= height else height
			x0 = (x-PADDING) if x-PADDING>=0 else 0
			x1 = (x+w+PADDING) if (x+w+PADDING) <= width else width
			sub_img = image[y0:y1, x0:x1]
			sub_img = cv2.resize(sub_img, (MIN_FACE_SIZE+PADDING, MIN_FACE_SIZE+PADDING))
			filename_ = os.path.splitext(filename)[0]
			if num_faces > 1:
				filename_ += "-"+str(i)
			cv2.imwrite(filename_+".jpg",sub_img)
	except Exception as e:
		print(e)
		print("Cannot process "+filename)

try:
	os.mkdir("/home/biagio/Documents/cubeyou/data/faces_64_")
except Exception:
	None
os.chdir("/home/biagio/Documents/cubeyou/data/faces_64_")

images = os.listdir("/home/biagio/Documents/cubeyou/data/training_set")
for img in images:
	detect_faces(img, "/home/biagio/Documents/cubeyou/data/training_set/"+img)

#images = os.listdir("C:/data/PainterByNumbers/train")
#for img in images:
#	detect_faces(img, "C:/data/PainterByNumbers/train/"+img)

#images = os.listdir("C:/data/PainterByNumbers/test")
#for img in images:
#	detect_faces(img, "C:/data/PainterByNumbers/test/"+img)
