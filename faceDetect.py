import cv2
#https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  #creating cascade classifier object

#read image
image = cv2.imread("groupfoto.jpeg")

# make duplicate of image and store as greyscale image
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# find face and give me the coordinates of the face as numpy array

face = faceCascade.detectMultiScale(grayImage,
    scaleFactor = 1.05,
    minNeighbors = 5)

#use the value of face and draw a rectangle in the image
green = (0,255,0)
#using for loop because there can be more than one faces in the image
# x, y = starting point of the face
for x, y, w, h in face:
    image = cv2.rectangle(image, (x,y), (x+w, y+h), green, 2)
cv2.imshow("detected face", image)

cv2.imwrite('detectedface.jpeg',image)
# cv2.imshow("newimage",grayImage)


cv2.waitKey(0)
