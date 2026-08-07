import os
import numpy as np
from PIL import Image

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import joblib


# ==========================================================
# SETTINGS
# ==========================================================

TRAIN_PATH = "PlantVillage/train"
VAL_PATH = "PlantVillage/val"

MODEL_PATH = "model/leaf_disease_model.pkl"

IMAGE_SIZE = (64, 64)


# ==========================================================
# LOAD DATA FUNCTION
# ==========================================================

def load_images(dataset_path):

    X = []
    y = []

    print()
    print("==========================================")
    print("Loading:", dataset_path)
    print("==========================================")

    if not os.path.exists(dataset_path):

        print("ERROR: Folder not found:")
        print(dataset_path)

        return np.array([]), np.array([])


    # Get every class folder automatically

    classes = sorted(
        [
            folder
            for folder in os.listdir(dataset_path)
            if os.path.isdir(
                os.path.join(dataset_path, folder)
            )
        ]
    )


    print("Total classes found:", len(classes))
    print()


    for class_name in classes:

        folder_path = os.path.join(
            dataset_path,
            class_name
        )

        image_count = 0


        for file_name in os.listdir(folder_path):

            if not file_name.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                continue


            file_path = os.path.join(
                folder_path,
                file_name
            )


            try:

                image = Image.open(
                    file_path
                )

                image = image.convert(
                    "RGB"
                )

                image = image.resize(
                    IMAGE_SIZE
                )

                image_array = np.array(
                    image
                )

                image_array = (
                    image_array / 255.0
                )

                image_array = (
                    image_array.flatten()
                )

                X.append(
                    image_array
                )

                y.append(
                    class_name
                )

                image_count += 1


            except Exception as e:

                print(
                    "Skipping:",
                    file_name,
                    "|",
                    e
                )


        print(
            f"{class_name}: {image_count} images"
        )


    return (
        np.array(X),
        np.array(y)
    )


# ==========================================================
# START
# ==========================================================

print()
print("==========================================")
print("   AgriVision AI Leaf Disease Training")
print("==========================================")


# ==========================================================
# LOAD TRAINING DATA
# ==========================================================

X_train, y_train = load_images(
    TRAIN_PATH
)


if len(X_train) == 0:

    print()
    print("ERROR: No training images found.")
    print("Check PlantVillage/train")

    exit()


# ==========================================================
# LOAD VALIDATION DATA
# ==========================================================

X_val, y_val = load_images(
    VAL_PATH
)


if len(X_val) == 0:

    print()
    print("ERROR: No validation images found.")
    print("Check PlantVillage/val")

    exit()


# ==========================================================
# DATA INFORMATION
# ==========================================================

print()
print("==========================================")
print("DATASET INFORMATION")
print("==========================================")

print(
    "Training images:",
    len(X_train)
)

print(
    "Validation images:",
    len(X_val)
)

print(
    "Number of classes:",
    len(np.unique(y_train))
)

print(
    "Feature size:",
    X_train.shape[1]
)


# ==========================================================
# CREATE MODEL
# ==========================================================

print()
print("==========================================")
print("Training Random Forest model...")
print("==========================================")


model = RandomForestClassifier(

    n_estimators=100,

    random_state=42,

    n_jobs=-1

)


# ==========================================================
# TRAIN MODEL
# ==========================================================

model.fit(
    X_train,
    y_train
)


print()
print("Training completed!")


# ==========================================================
# VALIDATION
# ==========================================================

print()
print("Testing model...")


y_pred = model.predict(
    X_val
)


accuracy = accuracy_score(
    y_val,
    y_pred
)


print()
print("==========================================")
print(
    f"Leaf Disease Model Accuracy: "
    f"{accuracy * 100:.2f}%"
)
print("==========================================")


# ==========================================================
# SAVE MODEL
# ==========================================================

os.makedirs(
    "model",
    exist_ok=True
)


joblib.dump(
    model,
    MODEL_PATH
)


print()
print("==========================================")
print("MODEL SAVED SUCCESSFULLY")
print("==========================================")

print(
    "Location:",
    MODEL_PATH
)

print()
print("All PlantVillage classes are now included.")