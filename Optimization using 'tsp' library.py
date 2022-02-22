#  This program utilizes the 'tsp' library for solving the
#  path optimisation problem

# Thereby most optimal path (shortest) is found!


import numpy as np
import cv2
import math
import pytesseract
import tsp

img = cv2.imread('Task_2.png')

# -----------------60% shrink to the image-------------------#
img = cv2.resize(img, None, fx=0.6, fy=0.6)

# -------------------Circle Detection------------------------#

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)

circles = cv2.HoughCircles(gray,
                           cv2.HOUGH_GRADIENT,
                           1, 10,
                           param1=15, param2=6,
                           minRadius=2, maxRadius=10)

detected_circles = np.uint16(np.around(circles))

# ------------Text Detection of R or G or B-------------------#

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
cap = cv2.VideoCapture(0)
while True:
    ret, source = cap.read()

    # making it suitable for tesseract by converting BGR to RGB
    suit = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

    # -----configuration to input only one character-----#
    # -------------which is 'R' or 'G' or 'B'------------#
    config = '--psm 10 --oem 3 -c tessedit_char_whitelist=RGB'

    ip = str(pytesseract.image_to_string(suit, config=config))
    print(ip)

    cv2.imshow('R-or-G-or-B scanner', source)

    if ip[0] == 'R' or ip[0] == 'B' or ip[0] == 'G':
        break
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
cap.release()
ch = ip[0]

n = 0
list_pt = []

for (x, y, r) in detected_circles[0, :]:

    # ----_blue_----#
    if ch == 'B':
        if img[y, x][0] > 175:  # blue
            print(x, y, r)
            n += 1
            list_pt.append([x, y])

    # ----_green_----#
    if ch == 'G':
        if img[y, x][1] > 175:  # green
            print(x, y, r)
            n += 1
            list_pt.append([x, y])

    # ----_red_----#
    if ch == 'R':
        if img[y, x][2] > 175:  # red
            print(x, y, r)
            n += 1
            list_pt.append([x, y])

print(n)

# ---CONSTRUCTION OF MATRIX FOR HAMILTONIAN PATH DETERMINATION----#

ar = []
for i in range(n):
    ar.append([])
    for j in range(n):
        x1 = float(list_pt[i][0])
        x2 = float(list_pt[j][0])
        y1 = float(list_pt[i][1])
        y2 = float(list_pt[j][1])
        X = x1 - x2
        Y = y1 - y2
        dist = math.sqrt(pow(X, 2) + pow(Y, 2))
        ar[i].append(dist)

r = range(len(ar))

shortestpath = {(i, j): ar[i][j] for i in r for j in r}

a, b = tsp.tsp(r, shortestpath)

path_length = a - ar[b[-1]] [b[0]]
# need not return to initial point as in TSP
# hence subtracting distance of last tour (path)

print(path_length, b)
                                   

##########################################
#########~~~~~~~~~~~~~~~~~~~~~~~##########
#########    DYNAMICS OF BOT    ##########
#########~~~~~~~~~~~~~~~~~~~~~~~##########
##########################################

# ---initial position of bot---#
x = float(list_pt[0][0])
y = float(list_pt[0][1])

# ------time instant------#
dt = 0.001  # <---- in seconds

delay = int(1000 * dt)  # <----in milliseconds

# ------PID controller parameters------#
Kp = 2.4
Ki = 0.4
Kd = 0.6


# ----Function to find angle----#
def findAngle(x, targetX, y, targetY):
    if x - targetX == 0:
        return math.pi / 2
    else:
        specs = math.atan((y - targetY) / (x - targetX))

        if y - targetY >= 0 and x - targetX < 0:
            return - specs

        if y - targetY > 0 and x - targetX > 0:
            return math.pi - specs

        if y - targetY < 0 and x - targetX < 0:
            return - specs

        if y - targetY < 0 and x - targetX > 0:
            return -(specs + math.pi)

        return

for i in range(len(b) - 1):

    targetX, targetY = float(list_pt[b[i + 1]][0]), float(list_pt[b[i + 1]][1])
    X = targetX - x
    Y = targetY - y
    e = math.sqrt(pow(X, 2) + pow(Y, 2))

    e_sum = 0
    e_der = 0
    e_dummy = e

    theta = findAngle(x, targetX, y, targetY)
    while (e > 1):
        copy=img.copy()
        # ----------error = distance--------#
        X = targetX - x
        Y = targetY - y
        e = math.sqrt(pow(X, 2) + pow(Y, 2))

        e_sum += e * dt
        e_der = (e - e_dummy) / dt
        e_dummy = e

        # actuating signal
        v = Kp * e + Ki * e_sum + Kd * e_der

        x += v * math.cos(theta) * dt
        y -= v * math.sin(theta) * dt

        x1 = int(x)
        y1 = int(y)

        #trail        
        cv2.circle(img, (x1, y1), 1, (0, 255, 255 ), -1)

        #bot
        cv2.circle(copy, (x1, y1), 4, (100, 50, 255 ), -1)
        cv2.imshow('output', copy)

        cv2.waitKey(delay)

cv2.waitKey(0)
cv2.destroyAllWindows()
