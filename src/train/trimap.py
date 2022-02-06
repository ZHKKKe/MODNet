import numpy as np
import cv2

def makeEdgeMask(mask, width):
    kernel = np.ones((width,width), np.uint8)

    erosion = cv2.erode(mask, kernel, iterations = 1)
    dilation = cv2.dilate(mask, kernel, iterations = 1)

    return dilation - erosion

def makeTrimap(mask, width = 5):
   edgeMask = makeEdgeMask(mask, width)
   trimap = mask.astype(np.float)
   trimap[edgeMask == 1] = 0.5
   return trimap