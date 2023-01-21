import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data from the "data/creditcard/transactions" folder
data = pd.read_csv("data/creditcard/transactions.csv")

# Split the data into training and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Define the features and target for the model
X_train = train_data.drop(columns=["Class"])
y_train = train_data["Class"]
X_test = test_data.drop(columns=["Class"])
y_test = test_data["Class"]

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# reshape the data to be fed into a CNN
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Create a sequential model with TensorFlow
model = tf.keras.Sequential()

# Add a convolutional layer with 32 filters and a kernel size of 3x3
model.add(tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))

# Add a max pooling layer with a pool size of 2x2
model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

# Add a convolutional layer with 64 filters and a kernel size of 3x3
model.add(tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu'))

# Add a max pooling layer with a pool size of 2x2
model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

# Flatten the output from the convolutional layers
model.add(tf.keras.layers.Flatten())

# Add a fully connected layer with 128 units and ReLU activation
model.add(tf.keras.layers.Dense(128, activation='relu'))

# Add a dropout layer to prevent overfitting
model.add(tf.keras.layers.Dropout(0.2))

# Add a fully connected layer with 1 unit and sigmoid activation
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Compile the model with binary crossentropy loss and Adam optimizer
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test accuracy:", test_acc)

# Use the model to make predictions on new data
predictions = model.predict(X_test)

