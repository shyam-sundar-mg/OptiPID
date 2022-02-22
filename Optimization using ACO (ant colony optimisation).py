#  This program uses ANT COLONY OPTIMISATION
#  to find an optimal path (not essentially the shortest)

#  only Heuristic solution can be achieved


import numpy as np
import cv2
import math
import pytesseract
from numpy.random import choice

img = cv2.imread('Task_2.png')

# -----------------60% shrink to the image-------------------#
img = cv2.resize(img, None, fx=0.6, fy=0.6)

print(img.shape)

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


##########################################
#########~~~~~~~~~~~~~~~~~~~~~~~##########
#########   PATH OPTIMIZATION   ##########
#########~~~~~~~~~~~~~~~~~~~~~~~##########
##########################################
############       by        #############
##########################################
#########~~~~~~~~~~~~~~~~~~~~~~~##########
#########       ANT COLONY      ##########
#########      OPTIMIZATION     ##########
#########        ALGORITHM      ##########
#########~~~~~~~~~~~~~~~~~~~~~~~##########
##########################################

n=len(list_pt)

dist = []
for i in range(n):
    dist.append([])
    for j in range(n):
        
        x1 = float(list_pt[i][0])
        x2 = float(list_pt[j][0])
        y1 = float(list_pt[i][1])
        y2 = float(list_pt[j][1])
        X = x1 - x2
        Y = y1 - y2
        distance = math.sqrt(pow(X, 2) + pow(Y, 2))
        dist[i].append(distance)

for i in dist:
    for j in i:
        print(int(j), end = ' ')
    print(end = '\n')

#-------INITIALIZATION OF PHEROMONE-------#
# initially all paths are equally probable
# assumption of equal pheromone is made
pheromone = []
for i in range(n):
    pheromone.append([])
    for j in range(n):
        pheromone[i].append(0.1)
        
def pathfn():
    
    #------------PROBABLE PATH-----------#
    
    path = []
    visited = set()
    visited.add(0)
    prev = 0
    for i in range(n - 1):
        move = ideal_choice(prev, pheromone, visited)        
        path.append((prev, move))
        prev = move        
        visited.add(move)
    phero_update(path)
    return path

def ideal_choice(prev, pher, vis):
    for i in vis:
        pher[prev][i] = 0
        
    #----------PROBABILITY-----------#   

    alpha = 0.6
    beta = 0.75

    numerator = []
    for i in range(n):
        numerator.append([])
        for j in range(n):
            if i==j:
                numerator[i].append(0) #append 0 or infinity, won't matter is code is perfect
                continue
            num = (pher[i][j] ** alpha) * ((1.0 / dist[i][j]) ** beta)
            numerator[i].append(num)
            
    sum_num = []
    for i in numerator:
        sum_num.append(sum(i))



    probability = []
    for i in range(n):
        probability.append([])
        for j in range(n):
            prolty = numerator[i][j] / sum_num[i]
            probability[i].append(prolty)

    #------RANDOM CHOICE WITH DIFFERENT PROBABILITIES------#
            #-------(weighted random choice)-------#
            
    move = choice(list(range(n)), size = 1, replace = False, p = probability[prev])[0]
    return move 

def phero_update(path):

    # L is the path length

    L = 0
    for (i,j) in path:
        L += dist[i][j]
        
    change = []
    for i in range(n):
        change.append([])
        for j in range(n):
            if i==j:
                change[i].append(0) #append 0 or infinity, won't matter is code is perfect
                continue
            change[i].append(1 / L)
    
    decay = 0.9
    
    for i in range(len(pheromone)):
        for j in range(len(pheromone)):
            if i==j:
                pheromone[i][j] = 0 #equal to 0 or infinity, won't matter is code is perfect
                continue
            pheromone[i][j] = decay * pheromone[i][j] + change [i][j]

# supereturn function is not compulsory
# (but for using the same code of bot, I've added this fn)
def supereturn(path):
    L = 0
    order = []
    for (i,j) in path:
        L += dist[i][j]
        order.append(i)
    else:
        order.append(j)
    return (L,order)

min = (485**2 + 1123**2) #(width x height = 485 x 1123) 
if n > 12:
    iterations = 10000
elif n > 8:
    iterations = 5000
else:
    iterations = 1000
for i in range(iterations):
    path = pathfn()
    print(supereturn(path))
    if supereturn(path)[0] < min:
        shortest_path = supereturn(path)[1]
        min = supereturn(path)[0]

print("\nHeuristic Minima")
a,b =  min, shortest_path
print(b)
print(min, shortest_path)

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
















