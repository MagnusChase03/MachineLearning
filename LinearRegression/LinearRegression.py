# Gets cordinates from file
def getData(file):
    f = open(file, "r")
    data = f.read()
    f.close()

    x = []
    y = []
    for line in data.split("\n"):
        cords = line.split(" ")
        x.append(float(cords[0]))
        y.append(float(cords[1]))

    cords = []
    cords.append(x)
    cords.append(y)
    
    return cords

# Caclulates best fit slope
def calculateSlope(x, y):
    n = len(x)

    xySum = 0
    xSum = 0
    xxSum = 0
    ySum = 0
    for i in range(0, n):
        xySum += x[i] * y[i]
        xSum += x[i]
        ySum += y[i]
        xxSum += x[i] * x[i]

    numerator = (n * xySum) - (xSum * ySum)
    denomenator = (n * xxSum) - (xSum * xSum)

    m = numerator / denomenator

    return m

# Caculates best fit intercept
def calculateIntercept(x, y, m):
    n = len(x)

    xSum = 0
    ySum = 0
    for i in range(0, n):
        xSum += x[i]
        ySum += y[i]

    numerator = ySum - (m * xSum)
    denomenator = n

    b = numerator / denomenator

    return b

def main():
    cords = getData("cords.txt")
    slope = calculateSlope(cords[0], cords[1])
    intercept = calculateIntercept(cords[0], cords[1], slope)
    print("y = %sx + %s" % (slope, intercept))

main()