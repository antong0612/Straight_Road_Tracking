import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import average

image = cv2.imread("test_image_2.png")
#================================Các hằng số=======================================#
#axis = 0: y axis
#axis = 1: x axis

width = image.shape[1]
height = image.shape[0]
#Kiểm tra kích thước của ảnh
#print(image.shape)
lane_image = np.copy(image)
#Hai thanh điều hướng
left_line = (100, 200, 250, 200)                      #Thanh trái
right_line = (390, 200, 540, 200)                     #Thanh phải                                      

cv2.imshow("Origninal image", lane_image)

#==============================Kết thúc hằng số====================================#

#================================Các hàm xử lý=====================================#

def canny(image):
    """
    Hàm tiền xử lý:
    Yêu cầu:
    +Chuyển ảnh RGB về Gray
    +Khử nhiễu
    +Sử dụng thuật toán Canny của openCV để nhận diện các cạnh
    """
    gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny_image = cv2.Canny(blur, 50, 150)
    #cv2.imshow("Canny", canny_image)
    return canny_image

def region_of_interest(image):
    """
    Hàm cắt ảnh
    Yêu cầu:
    +Tạo đa giác để cắt ảnh giữ lại các đường kẻ trên mặt đường
    """
    #print("Tao da giac chua phan anh can xu ly")
    polygons = np.array([[(0, 210), (0, height), (width, height), (width, 210), (320, 130)]])
    #print(polygons)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(canny_image, mask)
    #cv2.imshow("Mask", mask)
    return masked_image

def display_line(lines, image):
    """
    Hàm vẽ đoạn thẳng
    Yêu cầu:
    +Lấy tọa độ (x1, y1, x2, y2) từ một line trong mảng lines
    +Từ đó kẻ các đoạn thẳng có điểm đầu điểm cuối vào image 
    """
    lines_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(lines_image, (x1, y1), (x2, y2), (255,0,0), 2)
    #cv2.imshow("Lines Image", lines_image)
    return lines_image

def intercept_of_lines(line1, line2):
    """
    Hàm tìm giao điểm của hai đoạn thẳng cho trước
    Yêu cầu:
    +Nếu hai đoạn thẳng trùng nhau thì trả về trung điểm của đoạn thứ 2 (đoạn ta lấy tham chiếu)
    +Viết phương trình đoạn thẳng y = ax + b cho mỗi đoạn
    +Giao điểm tìm được phải nằm trong hai đoạn thẳng nói trên nếu không sẽ trả về None
    """
    x1, y1, x2, y2 = line1
    X1, Y1, X2, Y2 = line2
    if (x1 == x2) or (X1 == X2):
        return None
    #Tìm hệ số của phương trình đường thẳng
    a1 = (y2 - y1) / (x2 - x1)
    b1 = y1 - a1*x1
    a2 = (Y2 - Y1) / (X2 - X1)
    b2 = Y1 - a2*X1
    x, y = 0, 0
    if (a1 == a2):
        x, y = ((line2[0]+line2[1])/2, (line2[2]+line2[3])/2) 
    else:
        x = (b2-b1)/(a1-a2)
        y = a2*x + b2;
    if (x <= max(X1, X2) and x >= min(X1, X2)) and (y <= max(y1, y2) and y >= min(y1, y2)):
        return (x, y)
    return None

def display_intercepts(left_line, right_line, lines, image):
    combo_image = image
    left_points = []
    right_points = []
    if lines is not None:
        for line in lines:
            line_ = line.reshape(4)
            left_point = intercept_of_lines(line_, left_line)
            right_point = intercept_of_lines(line_, right_line)
            if left_point != None:
                left_points.append(left_point)
            if right_point != None:
                right_points.append(right_point)
    left_points = np.array(left_points, dtype=np.int32)
    #print(left_points)
    right_points = np.array(right_points, dtype = np.int32)
    #print(right_points)
    if left_points.size is not 0:
        averaged_left_point = np.mean(left_points, axis = 0, dtype = np.int32)
        cv2.circle(combo_image, averaged_left_point, 1, (0, 0, 255), 2)
        print(averaged_left_point)
    if right_points.size is not 0:
        averaged_right_point = np.mean(right_points, axis = 0, dtype = np.int32)
        print(averaged_right_point)
        cv2.circle(combo_image, averaged_right_point, 1, (0, 0, 255), 2)
    
    #cv2.imshow("Points Image", combo_image)
    return combo_image


#======================================Chạy code=============================================#

canny_image = canny(lane_image)
cropped_image = region_of_interest(canny_image)
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/200, 100, np.array([]), minLineLength = 40, maxLineGap = 5)
lines_image = display_line(lines, lane_image)
combo_image = cv2.addWeighted(lane_image, 0.8, lines_image, 1, 1)
cv2.line(combo_image, (100, 200), (250, 200), (23, 238, 253), 1)
cv2.line(combo_image, (390, 200), (540, 200), (23, 238, 253), 1)
combo_image = display_intercepts(left_line, right_line, lines, combo_image)
cv2.imshow("XXX", combo_image)
cv2.waitKey(0)

