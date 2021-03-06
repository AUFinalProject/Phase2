# Authors Alexey Titov and Shir Bentabou
# Version 5.0
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


# this function convert pdf file to jpg file
def convert(dirpdf):
    # dir of folder and filter for pdf files
    files = [
        f for f in os.listdir(dirpdf) if os.path.isfile(
            os.path.join(
                dirpdf,
                f))]
    files = list(filter(lambda f: f.endswith(('.pdf', '.PDF')), files))

    # variables for print information
    cnt_files = len(files)
    i = 0
    for filepdf in files:
        try:
            filename = os.path.join(dirpdf, filepdf)
            with tempfile.TemporaryDirectory() as path:
                images_from_path = convert_from_path(
                    filename, output_folder=path, last_page=1, first_page=0)

                base_filename = os.path.splitext(
                    os.path.basename(filename))[0] + '.jpg'
                save_dir = 'IMAGES'

                # save image
                for page in images_from_path:
                    page.save(os.path.join(save_dir, base_filename), 'JPEG')
            i += 1

            # show an update every 50 images
            if (i > 0 and i % 50 == 0):
                print("[INFO] processed {}/{}".format(i, cnt_files))
        except Exception:
            print(filepdf)
            # always keep track the error until the code has been clean
            print("[!] Convert PDF to JPEG")
            continue
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


# this function detect blur
def detect_image_blur(imgPath):
    try:
        image = cv2.imread(imgPath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(image, cv2.CV_64F).var()
        if (score < 110):
            detect = {score}
            return detect
        else:
            detect = {score}
            return detect
    except Exception:
        print(imgPath)
        detect = {0}
        return detect


if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
                    help="path to input dataset")
    ap.add_argument("-k", "--neighbors", type=int, default=5,
                    help="# of nearest neighbors for classification")
    ap.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=-
        1,
        help="# of jobs for k-NN distance (-1 uses all available cores)")
    args = vars(ap.parse_args())
    # define the name of the directory to be created
    path = "IMAGES"
    try:
        os.mkdir(path)
    except OSError:
        print(
            "[!] Creation of the directory %s failed, maybe the folder is exist" %
            path)
    else:
        print("[*] Successfully created the directory %s " % path)
    arg = os.path.join(os.getcwd(), args["dataset"])
    result = True #convert(arg)
    if (result):
        print("[*] Succces convert pdf files")
    else:
        print("[!] Whoops. something wrong dude. enable err var to track it")
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

        # detect blur
        blur = detect_image_blur(imagePath)

        hist = list(hist) + list(blur)
        hist = np.array(hist)

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
    (trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
        features, labels, test_size=0.20, random_state=42)

    # train and evaluate a k-NN classifer on the histogram
    # representations
    print("[INFO] evaluating histogram accuracy...")
    # Algorithm used to compute the nearest neighbors:
    model = KNeighborsClassifier(
        algorithm='auto',
        n_neighbors=args["neighbors"],
        n_jobs=args["jobs"])
    #	‘ball_tree’ will use BallTree
    #	‘kd_tree’ will use KDTree
    #	‘brute’ will use a brute-force search.
    #	‘auto’ will attempt to decide the most appropriate algorithm based on the values passed to fit method.

    model.fit(trainFeat, trainLabels)

    # Only accuracy
    acc = model.score(testFeat, testLabels)
    print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))

    # precision    recall	f1-score   support
    predictions = model.predict(testFeat)
    # show a final classification report demonstrating the accuracy of the
    # classifier
    print("EVALUATION ON TESTING DATA")
    print(classification_report(testLabels, predictions))
