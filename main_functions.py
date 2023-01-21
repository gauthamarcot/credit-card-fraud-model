import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def split_data(data, test_size=0.2, random_state=42):
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    return train_data, test_data

def normalize_data(train_data, test_data):
    X_train = train_data.drop(columns=["Class"])
    y_train = train_data["Class"]
    X_test = test_data.drop(columns=["Class"])
    y_test = test_data["Class"]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, y_train, X_test, y_test

def reshape_data(X_train, X_test):
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    return X_train, X_test

def create_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train, epochs=10, batch_size=32)

def evaluate_model(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("Test accuracy:", test_acc)
    
def predict(model, X_test):
    predictions = model.predict(X_test)
    # Do something with the predictions
    return predictions

def main():
    filepath = "data/creditcard/transactions.csv"
    data = load_data(filepath)
    train_data, test_data = split_data(data)
    X_train, y_train, X_test, y_test = normalize_data(train_data, test_data)
    X_train, X_test = reshape_data(X_train, X_test)
    model = create_model()
    train_model(model, X_train, y_train)
    evaluate_model(model, X_test, y_test)
    predictions = predict(model, X_test)

if __name__ == "__main__":
    main()
