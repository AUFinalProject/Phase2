# source http://optica.csic.es/papers/icpr2k.pdf
import cv2
threshold = 110

image = cv2.imread('example.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
score = cv2.Laplacian(image, cv2.CV_64F).var()

if score > threshold:
	print ("Not Blur")
else:
	print ("Blur")

print(score)

