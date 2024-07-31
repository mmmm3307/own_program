import cv2

img = cv2.imread('/home/ciiv04/program/own_program/Python/test/img/test3.bmp', cv2.IMREAD_GRAYSCALE)
cv2.imwrite('/home/ciiv04/program/own_program/Python/test/gray/gray3.jpg', img)