import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family":"hack","font.size":6})

def regoi(img, vertices):
    # Blank image of dim same as the image
    mask = np.zeros_like(img)
    # Retrieve the channel of the image
    # channel_count = img.shape[2]
    # Match color with same color channel count
    match_mask_color = 255
    # Mask everything other than ROI
    cv2.fillPoly(mask, vertices, match_mask_color)
    # return the image only where the mask pixels matches
    mask_img = cv2.bitwise_and(img, mask)
    return mask_img

def drawlines(img, lines):
    img = np.copy(img)
    blank_img = np.zeros((img.shape[0],img.shape[1],img.shape[2]), dtype = np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            #cv2.rectangle(blank_img, (x1, y1), (x2, y2), (0, 255, 0), -1)
            cv2.line(blank_img, (x1, y1), (x2, y2), (0, 0, 255), 5)
    img = cv2.addWeighted(img, 0.9, blank_img, 1, 0.0)
    return img


# for image
# img = cv2.imread("road.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def video(img):
    #print(img.shape)
    width = img.shape[1]
    height = img.shape[0]

    # Region of interest BottomLeft, Middle, BottomRight
    roi = [
        (-200, height),
        (width/2, 290),
        (width, height)
    ]

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gblur = cv2.GaussianBlur(img_gray,(5,5),0)


    canny = cv2.Canny(gblur, 100, 100)
    roi_img = regoi(canny, np.array([roi], np.int32))

    lines = cv2.HoughLinesP(roi_img, rho = 2, theta = np.pi/60, threshold = 100,
                            lines = None, minLineLength=50, maxLineGap = 50)


    img_lines = drawlines(img, lines)
    return img_lines

cap = cv2.VideoCapture("lane.mp4")
while(cap.isOpened()):
    ret, frame = cap.read()
    frame = video(frame)
    cv2.imshow("Lane Dection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
# plt.figure("Images", dpi=150, figsize=(6,5))
# plt.title("Lane Detection")
# plt.imshow(img_lines)
# plt.figure("Original", dpi=150, figsize=(6,5))
# plt.imshow(img)
# plt.figure("ROI", dpi=150, figsize=(6,5))
# plt.imshow(roi_img)
#plt.show()

cv2.destroyAllWindows()