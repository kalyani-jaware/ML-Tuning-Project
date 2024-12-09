import streamlit as st
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import zipfile
import tempfile

# Function to load images and split dataset
def load_dataset(data_dir, img_size):
    images = []
    labels = []
    class_names = os.listdir(data_dir)
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                try:
                    img = Image.open(img_path).resize(img_size)
                    images.append(np.array(img))
                    labels.append(label)
                except Exception as e:
                    st.warning(f"Could not load image {img_path}: {e}")

    return np.array(images), np.array(labels), class_names

# Streamlit App
def main():
    st.title("Image Classification with MobileNet Variants")

    # File uploader
    uploaded_file = st.file_uploader("Upload a zipped image dataset", type="zip")
    if uploaded_file:
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, "uploaded_data.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_file.read())
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            data_dir = temp_dir
            img_size = st.selectbox("Select Image Size", [(128, 128), (224, 224)], index=1)
            images, labels, class_names = load_dataset(data_dir, img_size)

            if images.size > 0:
                st.write(f"Dataset loaded successfully with {len(images)} images across {len(class_names)} classes.")
                st.write("Classes:", class_names)

                # Split data
                X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

                # Hyperparameter inputs
                st.sidebar.header("Hyperparameter Configuration")
                mobilenet_version = st.sidebar.selectbox("Select MobileNet Version", ["MobileNet", "MobileNetV2"])
                learning_rate = st.sidebar.slider(
                                        "Learning Rate (Approximate)",
                                        min_value=float(1e-6),
                                        max_value=float(1e-1),
                                        value=float(1e-3),
                                        step=float(1e-6),
                                        format="%.6f"
                                    )

                # # Text box for exact value
                # learning_rate = st.sidebar.text_input(
                #     "Learning Rate (Exact Input)",
                #     value=f"{lr_slider:.6f}"
                # )
                batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=1)
                epochs = st.sidebar.slider("Epochs", 1, 50, value=10)
                early_stop = st.sidebar.checkbox("Enable Early Stopping", value=True)

                # Data augmentation
                datagen = ImageDataGenerator(rescale=1.0/255, rotation_range=30, width_shift_range=0.2,
                                             height_shift_range=0.2, horizontal_flip=True, fill_mode="nearest")

                # Load selected MobileNet model
                if mobilenet_version == "MobileNet":
                    base_model = MobileNet(input_shape=img_size + (3,), include_top=False, weights="imagenet")
                else:
                    base_model = MobileNetV2(input_shape=img_size + (3,), include_top=False, weights="imagenet")

                base_model.trainable = False  # Freeze the base model

                # Build model
                model = models.Sequential([
                    base_model,
                    layers.GlobalAveragePooling2D(),
                    layers.Dense(128, activation="relu"),
                    layers.Dropout(0.5),
                    layers.Dense(len(class_names), activation="softmax")
                ])

                optimizer = Adam(learning_rate=learning_rate)
                model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

                # Callbacks
                callbacks = []
                if early_stop:
                    callbacks.append(EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True))

                # Training
                if st.button("Train Model"):
                    with st.spinner("Training in progress..."):
                        history = model.fit(
                            datagen.flow(X_train, y_train, batch_size=batch_size),
                            validation_data=(X_val / 255.0, y_val),
                            epochs=epochs,
                            callbacks=callbacks
                        )

                    st.success("Training completed!")

                    # Display metrics
                    st.write("Final Training Accuracy:", history.history["accuracy"][-1])
                    st.write("Final Validation Accuracy:", history.history["val_accuracy"][-1])

                    # Evaluate model
                    y_pred = model.predict(X_val / 255.0)
                    y_pred_classes = np.argmax(y_pred, axis=1)

                    # Classification Report
                    report = classification_report(y_val, y_pred_classes, target_names=class_names, output_dict=True)
                    st.write("Classification Report:")
                    st.dataframe(report)

                    # Confusion Matrix
                    cm = confusion_matrix(y_val, y_pred_classes)
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
                    plt.ylabel('True Labels')
                    plt.xlabel('Predicted Labels')
                    st.pyplot(fig)

                    # Save the model
                    if st.checkbox("Save Model"):
                        model_name = st.text_input("Enter model name:", value="mobilenet_model.h5")
                        model.save(model_name)
                        st.success(f"Model saved as {model_name}")

if __name__ == "__main__":
    main()
