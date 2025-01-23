import cv2

imgGraphics = cv2.imread('graphics.png')
if imgGraphics is None:
    print("Error: Image file could not be loaded.")
else:
    print("Image loaded successfully.")
