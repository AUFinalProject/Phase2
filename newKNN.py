# Authors Alexey Titov and Shir Bentabou
# Version 4.0
# Date 04.2019

# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report			
from pdf2image import convert_from_path
from PyPDF2 import PdfFileReader
from imutils import paths
import numpy as np
import pytesseract                                       
import argparse
import imutils
import cv2
import os
import tempfile
import sys


# dictionary for translate PDF language to tessaract language
lan_lst = {
	"en-us" : "eng",	"en" : "eng",	"en-za" : "eng",	"en-gb" : "eng",	"en-in" : "eng",
	"es-co" : "spa",	"es" : "spa",	"de-de" : "deu",	"fr-fr" : "fra",	"fr-ca" : "fra"
}

# dictionary for /Root/Lang 1 - except; 2 - a file have not /Root/Lang; 3 - /Root/Lang = ''; 4 - language
ans_list = dict()

# this function update ans_list
def add_ans_list(save_dir, base_filename, filename):
	try:
		name = os.path.join(save_dir, base_filename)
		pdfFile = PdfFileReader(file(filename, 'rb'))
		catalog = pdfFile.trailer['/Root'].getObject()
		if catalog.has_key("/Lang"):
			lang = catalog['/Lang'].getObject()
			if (lang == ''):
				ans_list.update( {name : [3, 'None']} )
			else:
				lang = lang.lower()
				language = lan_lst.get(lang)
				ans_list.update( {name : [4, language]} )
		else:
			ans_list.update( {name : [2, 'None']} )
	except:
		ans_list.update( {name : [1, 'None']} )

# this function convert pdf file to jpg file
def convert(dirpdf):
	# dir of folder and filter for pdf files
	files = [f for f in os.listdir(dirpdf) if os.path.isfile(os.path.join(dirpdf, f))]
	files = list(filter(lambda f: f.endswith(('.pdf','.PDF')), files))
	
	# variables for print information
	cnt_files = len(files)
	i = 0
	for filepdf in files:
		try:
			filename = os.path.join(dirpdf, filepdf)			
			with tempfile.TemporaryDirectory() as path:
	     			images_from_path = convert_from_path(filename, output_folder=path, last_page=1, first_page =0)
 
			base_filename  =  os.path.splitext(os.path.basename(filename))[0] + '.jpg'     
			save_dir = 'IMAGES'
 			
			# save image
			for page in images_from_path:
			    page.save(os.path.join(save_dir, base_filename), 'JPEG')
			i += 1	
			
			# update ans_list
			add_ans_list(save_dir, base_filename, filename)

			# show an update every 50 images
			if (i > 0 and i % 50 == 0):
				print("[INFO] processed {}/{}".format(i, cnt_files))
		except Exception:
        		# always keep track the error until the code has been clean
        		print("[!] Convert PDF to JPEG")
        		return False
	return True

# this function extract color histogram for images
def extract_color_histogram(image, bins=(8, 8, 8)):
	# extract a 3D color histogram from the HSV color space using
	# the supplied number of `bins` per channel
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])
 
	# handle normalizing the histogram if we are using OpenCV 2.4.X
	if imutils.is_cv2():
		hist = cv2.normalize(hist)
 
	# otherwise, perform "in place" normalization in OpenCV 3 (I
	# personally hate the way this is done
	else:
		cv2.normalize(hist, hist)
 
	# return the flattened histogram as the feature vector
	return hist.flatten()

# this function read information from image
def extract_text_image(imgPath):
	# Read image from disk
	img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
	
	# Read /Root/Lang
	values = ans_list.get(imgPath)
	try: 
		if (values[0] == 4):
			langs = value[1]
			imagetext = pytesseract.image_to_string(img, lang = langs)
		else:		
			imagetext = pytesseract.image_to_string(img)
		extract = {values[0], 0, 0}		
		return extract
	except:
		extract = {values[0], 1, 0}		
		return extract

# this function detect blur
def detect_image_blur(imgPath):
	image = cv2.imread(imgPath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	score = cv2.Laplacian(image, cv2.CV_64F).var()
	if (score < 110):
		detect = {1, score}
		return detect
	else:
		detect = {0, score}
		return detect

if __name__ == "__main__":
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--dataset", required = True,
		help="path to input dataset")
	ap.add_argument("-k", "--neighbors", type = int, default = 1,
		help="# of nearest neighbors for classification")
	ap.add_argument("-j", "--jobs", type = int, default = -1,
		help="# of jobs for k-NN distance (-1 uses all available cores)")
	args = vars(ap.parse_args())
	# define the name of the directory to be created
	path = "IMAGES"
	try:
    		os.mkdir(path)
	except OSError:  
	    	print ("[!] Creation of the directory %s failed, maybe the folder is exist" % path)
	else:  
    		print ("[*] Successfully created the directory %s " % path)
	arg = os.path.join(os.getcwd(), args["dataset"])    	
	result = convert(arg)
	if (result):
        	print ("[*] Succces convert pdf files")
	else:
		print ("[!] Whoops. something wrong dude. enable err var to track it")
		sys.exit()
	# grab the list of images that we'll be describing
	print("[INFO] describing images...")
	imagePaths = list(paths.list_images("IMAGES"))
	# initialize the raw pixel intensities matrix, the features matrix,
	# and labels list
	features = []
	labels = []
	# loop over the input images
	for (i, imagePath) in enumerate(imagePaths):
		# load the image and extract the class label (assuming that our
		# path as the format: /path/to/dataset/{class}.{image_num}.jpg
		image = cv2.imread(imagePath)
		label = imagePath.split(os.path.sep)[-1].split(".")[0]
 
		# histogram to characterize the color distribution of the pixels
		# in the image
		hist = extract_color_histogram(image)
		hist = np.array(hist)
	
		# text from image
		txt = extract_text_image(imagePath)
		np.concatenate((hist, txt), axis=None)
 		
		# detect blur
		blur = detect_image_blur(imagePath)
		np.concatenate((hist, blur), axis=None)
				
		# update the raw images, features, and labels matricies,
		# respectively
		features.append(hist)
		labels.append(label)
 		# show an update every 50 images
		if (i > 0 and i % 50 == 0):
			print("[INFO] processed {}/{}".format(i, len(imagePaths)))

	# show some information on the memory consumed by the raw images
	# matrix and features matrix
	features = np.array(features)
	labels = np.array(labels)
	print("[INFO] features matrix: {:.2f}MB".format(
		features.nbytes / (1024 * 1000.0)))

	# partition the data into training and testing splits, using 75%
	# of the data for training and the remaining 25% for testing
	(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(features, labels, test_size=0.25, random_state=42)

	# train and evaluate a k-NN classifer on the histogram
	# representations
	print("[INFO] evaluating histogram accuracy...")
	model = KNeighborsClassifier(algorithm='auto', n_neighbors = args["neighbors"],
		n_jobs=args["jobs"])                                                       # Algorithm used to compute the nearest neighbors:
											   #	‘ball_tree’ will use BallTree
                                                                                           #	‘kd_tree’ will use KDTree
                                                                                           #	‘brute’ will use a brute-force search.
    										           #	‘auto’ will attempt to decide the most appropriate algorithm based on the values passed to fit method.

	model.fit(trainFeat, trainLabels)
	
	# Only accuracy	
#	acc = model.score(testFeat, testLabels)
#	print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))
	
	# precision    recall	f1-score   support
	predictions = model.predict(testFeat)
	# show a final classification report demonstrating the accuracy of the classifier
	print("EVALUATION ON TESTING DATA")
	print(classification_report(testLabels, predictions))

	# one image for next phases
	# grab the image and classify it
	imagePath = 'white.27.jpg'
	image = cv2.imread(imagePath)
	hist = extract_color_histogram(image)
	add_ans_list('', imagePath, 'white.27.pdf')
	txt = extract_text_image(imagePath)
	blur = detect_image_blur(imagePath)
	hist = np.array(hist)
	np.concatenate((hist, txt), axis=None)
	np.concatenate((hist, blur), axis=None)
	prediction = model.predict(hist.reshape(1, -1))[0]
	# show the prediction
	print("I think that pdf is: {}".format(prediction))
