import random
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Cleaner display
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

# Read height and weight data
heightWeightData = pd.read_csv("heightWeight.csv")

# Get max and min for random centroids
heightMin = heightWeightData["Height"].min()
heightMax = heightWeightData["Height"].max()
weightMin = heightWeightData["Weight"].min()
weightMax = heightWeightData["Weight"].max()

# Using two centroids
centroid = [(random.random() * (weightMax - weightMin)) + weightMin, 
            (random.random() * (heightMax - heightMin)) + heightMin]

centroid2 = [(random.random() * (weightMax - weightMin)) + weightMin, 
            (random.random() * (heightMax - heightMin)) + heightMin]

# Calculate mean squared error to each centroid and ajust
for i in range(0, 100):
    centroidSum = [0, 0]
    centroid2Sum = [0, 0]
    centroidAmount = 0
    centroid2Amount = 0

    for point in heightWeightData[["Weight", "Height"]].iloc:
        centroidError = math.pow((point["Weight"] - centroid[0] + point["Height"] - centroid[1]), 2)
        centroid2Error = math.pow((point["Weight"] - centroid2[0] + point["Height"] - centroid2[1]), 2)

        if centroidError < centroid2Error:
            centroidAmount += 1
            centroidSum[0] += point["Weight"]
            centroidSum[1] += point["Height"]

        else:
            centroid2Amount += 1
            centroid2Sum[0] += point["Weight"]
            centroid2Sum[1] += point["Height"]

    centroidSum[0] /= centroidAmount
    centroidSum[1] /= centroidAmount
    centroid2Sum[0] /= centroid2Amount
    centroid2Sum[1] /= centroid2Amount

    centroid[0] = centroidSum[0]
    centroid[1] = centroidSum[1]
    centroid2[0] = centroid2Sum[0]
    centroid2[1] = centroid2Sum[1]

plt.plot([centroid[0], centroid2[0]], [centroid[1], centroid2[1]], 'ro')

# Plot by cluster
for point in heightWeightData[["Weight", "Height"]].iloc:
        centroidError = math.pow((point["Weight"] - centroid[0] + point["Height"] - centroid[1]), 2)
        centroid2Error = math.pow((point["Weight"] - centroid2[0] + point["Height"] - centroid2[1]), 2)

        if centroidError < centroid2Error:
            plt.plot([point["Weight"]], [point["Height"]], 'bo')

        else:
            plt.plot([point["Weight"]], [point["Height"]], 'go')

plt.show()