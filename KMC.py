# Authors Alexey Titov and Shir Bentabou
# Version 1.3
# Date 05.2019

# import the necessary packages
# data analysis and manipulation libraries
import numpy as np
import pandas as pd
# visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
# machine learning libraries
# importing K-Means
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from pdf2image import convert_from_path
from PyPDF2 import PdfFileReader
from imutils import paths
import pytesseract
import argparse
import imutils
import cv2
import os
import tempfile
import sys
from time import time
#import itertools
try:
    # Python 3
    from itertools import zip_longest
except ImportError:
    # Python 2
    from itertools import izip_longest as zip_longest

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
                images_from_path = convert_from_path(filename, output_folder=path, last_page=1, first_page=0)

                base_filename = os.path.splitext(os.path.basename(filename))[0] + '.jpg'
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
    ap.add_argument(
        "-c",
        "--clusters",
        type=int,
        default=20,
        help="the number of clusters to form as well as the number of centroids to generate")
    ap.add_argument("-j", "--jobs", type=int, default=-1,
                    help="the number of jobs to use for the computation. ")
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
    result = convert(arg)  # True for test
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
    # Get the features data
    data = features
    print("[INFO] features matrix: {:.2f}MB".format(
        data.nbytes / (1024 * 1000.0)))

    # instantiating kmeans
    km = KMeans(
        algorithm='auto',
        copy_x=True,
        init='k-means++',
        max_iter=300,
        n_clusters=args["clusters"],
        n_init=10,
        n_jobs=args["jobs"])

    # KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300, n_clusters=args["clusters"], n_init=10, n_jobs=args["jobs"], precompute_distances='auto', random_state=None, tol=0.0001, verbose=0)

    print("Clustering sparse data with %s" % km)
    t0 = time()
    # km.fit(data)
    clusters = km.fit_predict(data)
    print("done in %0.3fs" % (time() - t0))
    print("")

    # Homogeneity metric of a cluster labeling given a ground truth.
    # A clustering result satisfies homogeneity if all of its clusters contain only data points which are members of a single class.
    # score between 0.0 and 1.0. 1.0 stands for perfectly homogeneous labeling
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))

    # Completeness metric of a cluster labeling given a ground truth.
    # A clustering result satisfies completeness if all the data points that are members of a given class are elements of the same cluster.
    # score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling
    print(
        "Completeness: %0.3f" %
        metrics.completeness_score(
            labels,
            km.labels_))

    # V-measure cluster labeling given a ground truth.
    # This score is identical to normalized_mutual_info_score with the 'arithmetic' option for averaging.
    # score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling
    print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))

    # Rand index adjusted for chance.
    # The Rand Index computes a similarity measure between two clusterings by considering all pairs of samples and counting pairs that are assigned in
    # the same or different clusters in the predicted and true clusterings.
    # Similarity score between -1.0 and 1.0. Random labelings have an ARI
    # close to 0.0. 1.0 stands for perfect match.
    print(
        "Adjusted Rand-Index: %.3f" %
        metrics.adjusted_rand_score(
            labels, km.labels_))

    # Compute the mean Silhouette Coefficient of all samples.
    # The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters.
    # Negative values generally indicate that a sample has been assigned to
    # the wrong cluster, as a different cluster is more similar.
    print(
        "Silhouette Coefficient: %0.3f" %
        metrics.silhouette_score(
            data,
            km.labels_,
            sample_size=1000))
    print("")

    # empty dictionary
    results = {}
    for x in range(args["clusters"]):
        # add item
        results[x] = [0, 0]

    # iterates over 3 lists and till all are exhausted
    for (l, c) in zip_longest(labels, clusters):
        if (l == 'white'):				# white
            results[c][0] = results[c][0] + 1
        else:						# mal
            results[c][1] = results[c][1] + 1
    print(results)
    # data to plot
    n_groups = len(results)
    means_white = []
    means_mal = []
    indexs = []
    for x in range(args["clusters"]):
        means_white.append(results[x][0])
        means_mal.append(results[x][1])
        indexs.append(x)

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(
        index,
        means_white,
        bar_width,
        alpha=opacity,
        color='g',
        label='white')
    rects2 = plt.bar(
        index + bar_width,
        means_mal,
        bar_width,
        alpha=opacity,
        color='r',
        label='mal')

    plt.xlabel('Cluster')
    plt.ylabel('Elements')
    plt.title('Result of K means clustering')
    plt.xticks(index + bar_width, indexs)
    plt.legend()

    plt.tight_layout()
    plt.show()
