import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# Load data
df = pd.read_csv("train.csv")

# Preprocess data
label_encoder = LabelEncoder()
df["category"] = label_encoder.fit_transform(df["category"])
y = df["category"]
X = df.drop(["category", "ID"], axis=1)

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model architecture
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))

# Compile model
model.compile(
    loss="binary_crossentropy", optimizer=Adam(lr=0.001), metrics=["accuracy"]
)

# Set early stopping criteria
early_stopping = EarlyStopping(
    monitor="val_loss", patience=5, verbose=1, mode="min", restore_best_weights=True
)

# Train model
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=[early_stopping],
)

# Load test data
df_test = pd.read_csv("test.csv")
X_test = df_test.drop("ID", axis=1)

# Make predictions on test data
y_pred = model.predict(X_test)
y_pred = np.round(y_pred).astype(int)

# Convert predictions back to original labels
y_pred = label_encoder.inverse_transform(y_pred.flatten())

# Save predictions to file
df_submission = pd.DataFrame({"ID": df_test["ID"], "category": y_pred})
df_submission.to_csv("submission.csv", index=False)
