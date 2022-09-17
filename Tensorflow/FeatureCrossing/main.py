import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from matplotlib import pyplot as plt

# Cleaner output
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

tf.keras.backend.set_floatx('float32')

# Load data
utdGradesData = pd.read_csv("Spring2022.csv")

# todo: Replace NaN with 0

featureCols = []

# Get all professor names and create one hot feature
profs = []
for name in utdGradesData["Instructor1"]:

    if len(name) > 0 and not name in profs:
        profs.append(name)

profFeature = tf.feature_column.categorical_column_with_vocabulary_list("Instructor1", profs)
profFeatureOneHot = tf.feature_column.indicator_column(profFeature)
featureCols.append(profFeatureOneHot)

# Get all classes and create one hot feature
classes = []
for num in utdGradesData["CatalogNumber"]:

    if not num in classes:
        classes.append(num)

classsFeature = tf.feature_column.categorical_column_with_vocabulary_list("CatalogNumber", classes)
classesFeatureOneHot = tf.feature_column.indicator_column(classsFeature)
featureCols.append(classesFeatureOneHot)

# Create layer from features
featureColsLayer = layers.DenseFeatures(featureCols)

# Create model
model = tf.keras.Sequential()
model.add(featureColsLayer)
model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.3),
                loss="mean_squared_error",
                metrics=[tf.keras.metrics.RootMeanSquaredError()])

# Train model
features = {"CatalogNumber": np.array(utdGradesData["CatalogNumber"]), "Instructor1": np.array(utdGradesData["Instructor1"])}
label = np.array(utdGradesData["A"])

history = model.fit(x=features, y=label, batch_size=1, epochs=1)
hist = pd.DataFrame(history.history)

print(hist["root_mean_squared_error"])
