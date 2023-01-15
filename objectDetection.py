import cv2
import numpy as np

img = cv2.imread("irodov cover.jpg", cv2.IMREAD_GRAYSCALE)  # queryiamge
cap = cv2.VideoCapture(0)
vid = cv2.VideoCapture("C:\\Users\\Argha Kamal Samanta\\OneDrive\\Desktop\\videoplayback.mp4")
# img2 = cv2.imread("Doremon.png", cv2.IMREAD_GRAYSCALE)
# Features
sift = cv2.SIFT_create()
kp_image, desc_image = sift.detectAndCompute(img, None)
desc_image = np.float32(desc_image)
# Feature matching
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)
while True:
    _, frame = cap.read()
    _, img2 = vid.read()
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # trainimage
    kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)
    desc_grayframe = np.float32(desc_grayframe)
    matches = flann.knnMatch(desc_image, desc_grayframe, k=2)
    good_points = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good_points.append(m)

    # Homography
    if len(good_points) > 10:
        query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        # matches_mask = mask.ravel().tolist()
        # Perspective transform
        matrix = np.float32(matrix)
        h, w = img.shape
        canvas = np.float32(np.full((h, w), 255))
        result1 = cv2.warpPerspective(canvas, matrix, (grayframe.shape[1], grayframe.shape[0]))
        result1 = np.uint8(result1)
        result1 = cv2.bitwise_or(grayframe, result1)
        img2 = cv2.resize(img2, (w, h))
        result2 = cv2.warpPerspective(img2, matrix, (grayframe.shape[1], grayframe.shape[0]),
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        result2 = cv2.bitwise_and(result1, result2)
        result1 = cv2.cvtColor(result1, cv2.COLOR_GRAY2BGR)
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)
        homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
        cv2.imshow("Homography", homography)
        # cv2.imshow("Final video", frame)
        cv2.imshow("Result", result2)
    else:
        cv2.imshow("Homography", grayframe)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
