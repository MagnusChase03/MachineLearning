import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

# Cleaner display
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

# Read inflation data
inflationData = pd.read_csv("inflation.csv")

# Create a model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.1),
                loss="mean_squared_error",
                metrics=[tf.keras.metrics.RootMeanSquaredError()])

# Train the model
model.fit(x=inflationData["Row"],
            y=inflationData["Goods"],
            batch_size=10,
            epochs=1000)

trainedWeight = model.get_weights()[0]
trainedBias = model.get_weights()[1]

# Graph settings
plt.xlabel("Row")
plt.ylabel("Goods percent inflation")

# Plot points
plt.scatter(inflationData["Row"], inflationData["Goods"])

# Show model
x0 = 0
y0 = trainedBias
x1 = 10
y1 = trainedBias + (trainedWeight * x1)
plt.plot([x0, x1], [y0, y1], c='r')

plt.show()