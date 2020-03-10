import numpy as np
import cv2
import imutils
import pytesseract
# Read the image file
image = cv2.imread('img23.jpg')

# Resize the image - change width to 500
image = imutils.resize(image, width=500)
img2 = image.copy()
# Display the original image
#cv2.imshow("Original Image", image)

# RGB to Gray scale conversion
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("1 - Grayscale Conversion", gray)

# Noise removal with iterative bilateral filter(removes noise while preserving edges)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
#cv2.imshow("2 - Bilateral Filter", gray)

# Find Edges of the grayscale image
edged = cv2.Canny(gray, 170, 200)
#cv2.imshow("4 - Canny Edges", edged)

# Find contours based on Edges
(new, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:100] #sort contours based on their area keeping minimum required area as '30' (anything smaller than this will not be considered)
NumberPlateCnt = None #we currently have no Number plate contour

# loop over our contours to find the best possible approximate contour of number plate
count = 0
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    if len(approx) == 4:  # Select the contour with 4 corners
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        if ar not in [0.95, 1.05]:
            NumberPlateCnt = approx
            img_new = img2[y:y + h, x:x + w]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imwrite("image_cropped.png", img_new)

            break

img_text = cv2.imread('image_cropped.png')
gray = cv2.cvtColor(img_text, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (15, 15), 0)
img_finale = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
txt = pytesseract.image_to_string(gray, config='tessedit_char_whitelist=0123456789 -psm 6 ')
print("License Plate Number : ", txt)
cv2.putText(image, txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Drawing the selected contour on the original image
cv2.drawContours(img2, [NumberPlateCnt], -1, (0,255,0), 3)
cv2.imshow("Final Image With Number Plate Detected", image)
cv2.imshow("image copy", img2)
cv2.imshow("image cropped", img_finale)
cv2.waitKey(0)