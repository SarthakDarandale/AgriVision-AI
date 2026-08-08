from flask import Flask, render_template, request, send_file
import csv
import html
import os
import random
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import requests
from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError

from crop_info import crop_info
from maharashtra_crop import maharashtra_crops

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# ==========================================================
# FLASK APP
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)

# Maximum upload size: 5 MB
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024


# ==========================================================
# PATHS
# ==========================================================

MODEL_DIR = BASE_DIR / "model"
SAVED_PREDICTIONS_DIR = BASE_DIR / "saved_predictions"
SAVED_MESSAGES_DIR = BASE_DIR / "saved_messages"
REPORTS_DIR = BASE_DIR / "reports"

CROP_MODEL_PATH = MODEL_DIR / "crop_model.pkl"

# Hugging Face leaf disease model
LEAF_MODEL_REPO = "Sarth1602/agrivision-leaf-disease-model"
LEAF_MODEL_FILENAME = "leaf_disease_model.pkl"

# FIX:
# This variable was missing in your previous app.py.
LEAF_MODEL_PATH = MODEL_DIR / LEAF_MODEL_FILENAME

HISTORY_FILE = SAVED_PREDICTIONS_DIR / "prediction_history.csv"
MESSAGES_FILE = SAVED_MESSAGES_DIR / "messages.csv"
PDF_PATH = REPORTS_DIR / "prediction_report.pdf"


# ==========================================================
# DIRECTORY SETUP
# ==========================================================

def ensure_directories():
    """Create folders required by the application."""

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SAVED_PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SAVED_MESSAGES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ==========================================================
# MODEL LOADING
# ==========================================================

def load_model(path, model_name):
    """
    Load a local joblib model and provide
    a useful error message if loading fails.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"{model_name} not found at:\n{path}"
        )

    try:
        return joblib.load(path)

    except Exception as exc:
        raise RuntimeError(
            f"Unable to load {model_name}: {exc}"
        ) from exc


def download_leaf_model():
    """
    Download the leaf disease model from Hugging Face
    if it is not already available locally.
    """

    ensure_directories()

    if LEAF_MODEL_PATH.exists():
        print(
            f"Leaf disease model found:\n"
            f"{LEAF_MODEL_PATH}"
        )
        return

    print("=" * 60)
    print("Leaf disease model not found.")
    print("Downloading from Hugging Face...")
    print(f"Repository: {LEAF_MODEL_REPO}")
    print(f"Filename: {LEAF_MODEL_FILENAME}")
    print("=" * 60)

    try:
        downloaded_path = hf_hub_download(
            repo_id=LEAF_MODEL_REPO,
            filename=LEAF_MODEL_FILENAME,
        )

        downloaded_path = Path(downloaded_path)

        # Copy the downloaded model into our project model folder
        import shutil

        shutil.copy2(
            downloaded_path,
            LEAF_MODEL_PATH,
        )

        print(
            "Leaf disease model downloaded successfully."
        )

    except Exception as exc:
        raise RuntimeError(
            "Unable to download the leaf disease model "
            "from Hugging Face.\n\n"
            f"Repository: {LEAF_MODEL_REPO}\n"
            f"File: {LEAF_MODEL_FILENAME}\n\n"
            f"Original error: {exc}"
        ) from exc


# ==========================================================
# STARTUP MODEL LOADING
# ==========================================================

ensure_directories()

# Crop model must already exist
model = load_model(
    CROP_MODEL_PATH,
    "Crop model",
)

# Download leaf model if necessary
download_leaf_model()

# Load leaf model
leaf_model = load_model(
    LEAF_MODEL_PATH,
    "Leaf disease model",
)


# ==========================================================
# LATEST PREDICTION
# ==========================================================

latest_prediction = None
latest_details = None
latest_inputs = None
latest_prediction_time = None


# ==========================================================
# FARMING TIPS
# ==========================================================

farming_tips = [
    "🌱 Test your soil before planting to improve crop yield.",
    "💧 Water crops early in the morning to reduce evaporation.",
    "🌾 Rotate crops every season to maintain soil fertility.",
    "🌿 Use organic compost to improve soil health.",
    "☀️ Avoid overwatering to prevent root diseases.",
    "🐞 Regularly inspect crops for pests and diseases.",
    "🌦️ Check the weather forecast before irrigating fields.",
    "🚜 Prepare the land properly before sowing seeds.",
    "🌱 Use certified quality seeds for better production.",
    "🧪 Apply fertilizers according to soil test results.",
    "🍂 Remove weeds regularly to reduce nutrient competition.",
    "🌾 Harvest crops at the right maturity stage.",
    "💦 Install drip irrigation to save water.",
    "🌳 Plant trees around farms to reduce soil erosion.",
    "🐝 Encourage pollinators like bees for better crop production.",
    "🌧️ Store rainwater for irrigation during dry seasons.",
    "🌍 Practice sustainable farming to protect the environment.",
    "🪱 Use vermicompost to increase soil nutrients naturally.",
    "☘️ Mulching helps retain soil moisture and control weeds.",
    "📅 Keep a record of farming activities for better planning.",
]


# ==========================================================
# WEATHER INFORMATION
# ==========================================================

WEATHER_CODES = {
    0: "Clear Sky",
    1: "Mainly Clear",
    2: "Partly Cloudy",
    3: "Cloudy",
    45: "Fog",
    48: "Dense Fog",
    51: "Light Drizzle",
    53: "Moderate Drizzle",
    55: "Heavy Drizzle",
    56: "Light Freezing Drizzle",
    57: "Heavy Freezing Drizzle",
    61: "Light Rain",
    63: "Moderate Rain",
    65: "Heavy Rain",
    66: "Light Freezing Rain",
    67: "Heavy Freezing Rain",
    71: "Light Snow",
    73: "Moderate Snow",
    75: "Heavy Snow",
    77: "Snow Grains",
    80: "Rain Showers",
    81: "Moderate Rain Showers",
    82: "Heavy Rain Showers",
    85: "Light Snow Showers",
    86: "Heavy Snow Showers",
    95: "Thunderstorm",
    96: "Thunderstorm with Hail",
    99: "Severe Thunderstorm",
}


WEATHER_ICONS = {
    0: "☀️",
    1: "🌤️",
    2: "⛅",
    3: "☁️",
    45: "🌫️",
    48: "🌫️",
    51: "🌦️",
    53: "🌦️",
    55: "🌧️",
    56: "🌧️",
    57: "🌧️",
    61: "🌦️",
    63: "🌧️",
    65: "🌧️",
    66: "🌧️",
    67: "🌧️",
    71: "❄️",
    73: "❄️",
    75: "❄️",
    77: "❄️",
    80: "🌦️",
    81: "🌧️",
    82: "🌧️",
    85: "🌨️",
    86: "🌨️",
    95: "⛈️",
    96: "⛈️",
    99: "⛈️",
}


# ==========================================================
# DISEASE INFORMATION
# ==========================================================

DISEASE_INFO = {

    "Apple___Apple_scab": {
        "name": "Apple Scab",
        "description": (
            "Apple scab is a fungal disease that causes "
            "dark lesions on apple leaves and fruit."
        ),
        "advice": (
            "Remove affected leaves and fallen plant "
            "material. Maintain good air circulation "
            "and avoid prolonged leaf wetness."
        ),
    },

    "Apple___Black_rot": {
        "name": "Apple Black Rot",
        "description": (
            "Black rot can cause dark lesions on apple "
            "leaves and fruit."
        ),
        "advice": (
            "Remove infected plant parts and maintain "
            "good orchard sanitation."
        ),
    },

    "Apple___Cedar_apple_rust": {
        "name": "Apple Cedar Apple Rust",
        "description": (
            "Cedar apple rust causes yellow-orange spots "
            "on apple leaves."
        ),
        "advice": (
            "Remove heavily affected leaves and improve "
            "air circulation around plants."
        ),
    },

    "Apple___healthy": {
        "name": "Healthy Apple Leaf",
        "description": (
            "The uploaded apple leaf appears healthy."
        ),
        "advice": (
            "Continue regular monitoring, proper watering "
            "and balanced crop nutrition."
        ),
    },

    "Blueberry___healthy": {
        "name": "Healthy Blueberry Leaf",
        "description": (
            "The uploaded blueberry leaf appears healthy."
        ),
        "advice": (
            "Continue regular crop monitoring and maintain "
            "proper soil moisture."
        ),
    },

    "Cherry_(including_sour)___Powdery_mildew": {
        "name": "Cherry Powdery Mildew",
        "description": (
            "Powdery mildew produces a white powder-like "
            "growth on leaves and shoots."
        ),
        "advice": (
            "Improve air circulation, avoid excessive "
            "humidity and remove severely affected "
            "plant parts."
        ),
    },

    "Cherry_(including_sour)___healthy": {
        "name": "Healthy Cherry Leaf",
        "description": (
            "The uploaded cherry leaf appears healthy."
        ),
        "advice": (
            "Continue regular monitoring and maintain "
            "good orchard hygiene."
        ),
    },

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "name": "Corn Gray Leaf Spot",
        "description": (
            "Gray leaf spot causes characteristic gray "
            "or brown lesions on corn leaves."
        ),
        "advice": (
            "Remove severely affected plant material and "
            "improve field airflow. Avoid prolonged "
            "leaf wetness."
        ),
    },

    "Corn_(maize)___Common_rust": {
        "name": "Corn Common Rust",
        "description": (
            "Common rust produces reddish-brown rust-colored "
            "pustules on corn leaves."
        ),
        "advice": (
            "Monitor affected plants and maintain good "
            "crop management. Use resistant varieties "
            "where available."
        ),
    },

    "Corn_(maize)___Northern_Leaf_Blight": {
        "name": "Corn Northern Leaf Blight",
        "description": (
            "Northern leaf blight produces elongated "
            "gray-green or brown lesions on corn leaves."
        ),
        "advice": (
            "Remove severely affected plant material and "
            "maintain good field sanitation."
        ),
    },

    "Corn_(maize)___healthy": {
        "name": "Healthy Corn Leaf",
        "description": (
            "The uploaded corn leaf appears healthy."
        ),
        "advice": (
            "Continue regular crop monitoring and maintain "
            "appropriate irrigation and nutrition."
        ),
    },

    "Grape___Black_rot": {
        "name": "Grape Black Rot",
        "description": (
            "Black rot causes brown lesions on grape leaves "
            "and can affect grape clusters."
        ),
        "advice": (
            "Remove infected plant material and improve "
            "air circulation around the vines."
        ),
    },

    "Grape___Esca_(Black_Measles)": {
        "name": "Grape Esca (Black Measles)",
        "description": (
            "Esca is a grapevine disease that can cause "
            "characteristic leaf symptoms and fruit damage."
        ),
        "advice": (
            "Remove severely affected plant material and "
            "maintain vineyard sanitation."
        ),
    },

    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "name": "Grape Leaf Blight",
        "description": (
            "Grape leaf blight causes spots and lesions "
            "on grape leaves."
        ),
        "advice": (
            "Remove affected leaves and improve air "
            "circulation within the vineyard."
        ),
    },

    "Grape___healthy": {
        "name": "Healthy Grape Leaf",
        "description": (
            "The uploaded grape leaf appears healthy."
        ),
        "advice": (
            "Continue regular vineyard monitoring and "
            "proper crop care."
        ),
    },

    "Orange___Haunglongbing_(Citrus_greening)": {
        "name": "Citrus Greening",
        "description": (
            "Citrus greening can cause yellowing and "
            "abnormal leaf development in citrus plants."
        ),
        "advice": (
            "Monitor affected plants carefully and consult "
            "an agricultural expert for appropriate management."
        ),
    },

    "Peach___Bacterial_spot": {
        "name": "Peach Bacterial Spot",
        "description": (
            "Bacterial spot can produce dark spots and "
            "lesions on peach leaves."
        ),
        "advice": (
            "Remove severely affected plant parts and "
            "maintain good orchard sanitation."
        ),
    },

    "Peach___healthy": {
        "name": "Healthy Peach Leaf",
        "description": (
            "The uploaded peach leaf appears healthy."
        ),
        "advice": (
            "Continue regular monitoring and proper "
            "orchard care."
        ),
    },

    "Pepper,_bell___Bacterial_spot": {
        "name": "Pepper Bacterial Spot",
        "description": (
            "Bacterial spot causes dark lesions on pepper "
            "leaves and fruit."
        ),
        "advice": (
            "Remove affected plant material and avoid "
            "overhead watering."
        ),
    },

    "Pepper,_bell___healthy": {
        "name": "Healthy Pepper Leaf",
        "description": (
            "The uploaded pepper leaf appears healthy."
        ),
        "advice": (
            "Continue regular monitoring and proper watering."
        ),
    },

    "Potato___Early_blight": {
        "name": "Potato Early Blight",
        "description": (
            "Early blight causes dark lesions and concentric "
            "ring patterns on potato leaves."
        ),
        "advice": (
            "Remove affected leaves, maintain good field "
            "sanitation and avoid prolonged leaf wetness."
        ),
    },

    "Potato___Late_blight": {
        "name": "Potato Late Blight",
        "description": (
            "Late blight can cause dark, water-soaked "
            "lesions on potato leaves."
        ),
        "advice": (
            "Remove infected plant material, improve air "
            "circulation and avoid overhead watering."
        ),
    },

    "Potato___healthy": {
        "name": "Healthy Potato Leaf",
        "description": (
            "The uploaded potato leaf appears healthy."
        ),
        "advice": (
            "Continue regular crop monitoring and proper "
            "crop care."
        ),
    },

    "Raspberry___healthy": {
        "name": "Healthy Raspberry Leaf",
        "description": (
            "The uploaded raspberry leaf appears healthy."
        ),
        "advice": (
            "Continue regular monitoring and maintain "
            "appropriate watering and nutrition."
        ),
    },

    "Soybean___healthy": {
        "name": "Healthy Soybean Leaf",
        "description": (
            "The uploaded soybean leaf appears healthy."
        ),
        "advice": (
            "Continue regular monitoring and proper "
            "crop management."
        ),
    },

    "Squash___Powdery_mildew": {
        "name": "Squash Powdery Mildew",
        "description": (
            "Powdery mildew produces a white powder-like "
            "growth on squash leaves."
        ),
        "advice": (
            "Improve air circulation and remove severely "
            "affected leaves."
        ),
    },

    "Strawberry___Leaf_scorch": {
        "name": "Strawberry Leaf Scorch",
        "description": (
            "Leaf scorch produces dark lesions and scorched "
            "areas on strawberry leaves."
        ),
        "advice": (
            "Remove severely affected leaves and maintain "
            "proper irrigation and field sanitation."
        ),
    },

    "Strawberry___healthy": {
        "name": "Healthy Strawberry Leaf",
        "description": (
            "The uploaded strawberry leaf appears healthy."
        ),
        "advice": (
            "Continue regular monitoring and proper crop care."
        ),
    },

    "Tomato___Bacterial_spot": {
        "name": "Tomato Bacterial Spot",
        "description": (
            "Bacterial spot causes small dark lesions on "
            "tomato leaves and fruit."
        ),
        "advice": (
            "Remove affected leaves and avoid overhead watering."
        ),
    },

    "Tomato___Early_blight": {
        "name": "Tomato Early Blight",
        "description": (
            "Early blight causes dark lesions with "
            "characteristic ring patterns on tomato leaves."
        ),
        "advice": (
            "Remove affected leaves, improve air circulation "
            "and avoid excess moisture."
        ),
    },

    "Tomato___Late_blight": {
        "name": "Tomato Late Blight",
        "description": (
            "Late blight can cause dark brown or black "
            "lesions on tomato leaves and other plant parts."
        ),
        "advice": (
            "Remove infected leaves and severely affected "
            "plant parts. Improve air circulation and avoid "
            "overhead watering."
        ),
    },

    "Tomato___Leaf_Mold": {
        "name": "Tomato Leaf Mold",
        "description": (
            "Leaf mold can cause yellow areas on the upper "
            "surface of tomato leaves and fungal growth underneath."
        ),
        "advice": (
            "Improve ventilation, reduce humidity and avoid "
            "prolonged leaf wetness."
        ),
    },

    "Tomato___Septoria_leaf_spot": {
        "name": "Tomato Septoria Leaf Spot",
        "description": (
            "Septoria leaf spot produces numerous small "
            "spots on tomato leaves."
        ),
        "advice": (
            "Remove affected leaves, maintain field sanitation "
            "and avoid overhead watering."
        ),
    },

    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "name": "Tomato Spider Mites",
        "description": (
            "Spider mites can cause speckling, yellowing and "
            "damage to tomato leaves."
        ),
        "advice": (
            "Inspect the underside of leaves and manage the "
            "infestation using appropriate agricultural "
            "pest-control practices."
        ),
    },

    "Tomato___Target_Spot": {
        "name": "Tomato Target Spot",
        "description": (
            "Target spot produces circular lesions on "
            "tomato leaves."
        ),
        "advice": (
            "Remove severely affected leaves and improve "
            "air circulation."
        ),
    },

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "name": "Tomato Yellow Leaf Curl Virus",
        "description": (
            "This viral disease can cause leaf curling, "
            "yellowing and stunted plant growth."
        ),
        "advice": (
            "Remove severely infected plants and manage "
            "insect vectors such as whiteflies with "
            "appropriate agricultural practices."
        ),
    },

    "Tomato___Tomato_mosaic_virus": {
        "name": "Tomato Mosaic Virus",
        "description": (
            "Tomato mosaic virus can cause mottled leaf "
            "patterns and reduced plant growth."
        ),
        "advice": (
            "Remove severely affected plants and maintain "
            "good tool and field sanitation."
        ),
    },

    "Tomato___healthy": {
        "name": "Healthy Tomato Leaf",
        "description": (
            "The uploaded tomato leaf appears healthy."
        ),
        "advice": (
            "Continue regular monitoring, proper watering "
            "and balanced crop nutrition."
        ),
    },
}


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def clean_text(value, default=""):
    """Convert a value to safe plain text."""

    if value is None:
        return default

    return str(value).strip()


def validate_crop_inputs(values):
    """Validate crop-prediction form values."""

    field_names = [
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall",
    ]

    parsed = {}

    for field in field_names:

        raw_value = values.get(
            field,
            "",
        ).strip()

        if raw_value == "":
            raise ValueError(
                f"Please enter {field}."
            )

        try:
            number = float(raw_value)

        except ValueError:
            raise ValueError(
                f"{field} must be a valid number."
            )

        if not np.isfinite(number):
            raise ValueError(
                f"{field} must be a valid finite number."
            )

        parsed[field] = number

    # Basic domain validation

    if (
        parsed["N"] < 0
        or parsed["P"] < 0
        or parsed["K"] < 0
    ):
        raise ValueError(
            "N, P and K values cannot be negative."
        )

    if (
        parsed["humidity"] < 0
        or parsed["humidity"] > 100
    ):
        raise ValueError(
            "Humidity must be between 0 and 100."
        )

    if (
        parsed["ph"] < 0
        or parsed["ph"] > 14
    ):
        raise ValueError(
            "pH must be between 0 and 14."
        )

    if parsed["rainfall"] < 0:
        raise ValueError(
            "Rainfall cannot be negative."
        )

    return parsed


def save_prediction_to_history(
    result,
    inputs,
):
    """Save crop prediction in CSV history."""

    ensure_directories()

    file_exists = HISTORY_FILE.exists()

    with HISTORY_FILE.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "Date",
                "Crop",
                "Nitrogen",
                "Phosphorus",
                "Potassium",
                "Temperature",
                "Humidity",
                "pH",
                "Rainfall",
            ])

        writer.writerow([
            datetime.now().strftime(
                "%d-%m-%Y %H:%M"
            ),
            result,
            inputs["N"],
            inputs["P"],
            inputs["K"],
            inputs["temperature"],
            inputs["humidity"],
            inputs["ph"],
            inputs["rainfall"],
        ])


def get_disease_info(prediction):
    """Return readable information for disease model class."""

    info = DISEASE_INFO.get(prediction)

    if info:
        return info

    return {
        "name": clean_text(
            prediction
        ).replace(
            "___",
            " - ",
        ),

        "description": (
            "The AI model detected this PlantVillage class."
        ),

        "advice": (
            "Monitor the crop regularly and consult "
            "an agricultural expert if symptoms continue."
        ),
    }


def weather_code_to_text(code):
    return WEATHER_CODES.get(
        code,
        "Unknown Weather",
    )


def weather_code_to_icon(code):
    return WEATHER_ICONS.get(
        code,
        "🌍",
    )


# ==========================================================
# HOME
# ==========================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        tip=random.choice(
            farming_tips
        ),
    )


# ==========================================================
# CROP PREDICTION PAGE
# ==========================================================

@app.route("/predict-page")
def predict_page():

    return render_template(
        "predict.html"
    )


# ==========================================================
# ABOUT
# ==========================================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# ==========================================================
# CONTACT
# ==========================================================

@app.route(
    "/contact",
    methods=["GET", "POST"],
)
def contact():

    if request.method == "GET":

        return render_template(
            "contact.html"
        )

    name = clean_text(
        request.form.get("name")
    )

    email = clean_text(
        request.form.get("email")
    )

    message = clean_text(
        request.form.get("message")
    )

    if not name or not email or not message:

        return render_template(
            "contact.html",
            error="Please fill in all the fields.",
        )

    if len(name) > 100:

        return render_template(
            "contact.html",
            error="Name is too long.",
        )

    if len(email) > 150:

        return render_template(
            "contact.html",
            error="Email is too long.",
        )

    if len(message) > 2000:

        return render_template(
            "contact.html",
            error=(
                "Message is too long. "
                "Maximum 2000 characters."
            ),
        )

    ensure_directories()

    file_exists = MESSAGES_FILE.exists()

    with MESSAGES_FILE.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "Date",
                "Name",
                "Email",
                "Message",
            ])

        writer.writerow([
            datetime.now().strftime(
                "%d-%m-%Y %H:%M"
            ),
            name,
            email,
            message,
        ])

    return render_template(
        "contact.html",
        success=(
            "Your message has been sent successfully. "
            "Thank you for contacting AgriVision AI!"
        ),
    )


# ==========================================================
# MESSAGES DASHBOARD
# ==========================================================

@app.route("/messages")
def messages():

    messages_data = []

    if MESSAGES_FILE.exists():

        try:

            with MESSAGES_FILE.open(
                "r",
                newline="",
                encoding="utf-8",
            ) as file:

                reader = csv.DictReader(
                    file
                )

                for row in reader:
                    messages_data.append(row)

        except Exception as exc:

            print(
                "Messages file error:",
                exc,
            )

    messages_data.reverse()

    return render_template(
        "messages.html",
        messages=messages_data,
    )


# ==========================================================
# LEAF DISEASE DETECTION
# ==========================================================

@app.route(
    "/disease",
    methods=["GET", "POST"],
)
def disease():

    if request.method == "GET":

        return render_template(
            "disease.html"
        )

    if "leaf_image" not in request.files:

        return render_template(
            "disease.html",
            error="Please select a leaf image.",
        )

    uploaded_file = request.files[
        "leaf_image"
    ]

    if (
        not uploaded_file
        or uploaded_file.filename == ""
    ):

        return render_template(
            "disease.html",
            error="Please select a leaf image.",
        )

    try:

        # Open image
        image = Image.open(
            uploaded_file
        )

        # Verify image
        image.verify()

        # Re-open after verify
        uploaded_file.stream.seek(0)

        image = Image.open(
            uploaded_file
        )

        # Convert to RGB
        image = image.convert(
            "RGB"
        )

        # Resize according to model input
        image = image.resize(
            (64, 64)
        )

        # Convert to NumPy
        image_array = np.array(
            image,
            dtype=np.float32,
        )

        # Normalize
        image_array = (
            image_array / 255.0
        )

        # Flatten
        image_array = (
            image_array.flatten()
        )

        # Add batch dimension
        image_array = image_array.reshape(
            1,
            -1,
        )

        # Prediction
        prediction = leaf_model.predict(
            image_array
        )[0]

        prediction = clean_text(
            prediction
        )

        info = get_disease_info(
            prediction
        )

        return render_template(
            "disease_result.html",
            prediction=info["name"],
            recommendation=info["advice"],
            description=info["description"],
        )

    except (
        UnidentifiedImageError,
        OSError,
    ):

        return render_template(
            "disease.html",
            error=(
                "Invalid image. Please upload a "
                "valid JPG, JPEG or PNG leaf image."
            ),
        )

    except Exception as exc:

        print(
            "Leaf prediction error:",
            exc,
        )

        return render_template(
            "disease.html",
            error=(
                "Unable to analyze the image. "
                "Please upload a valid JPG or PNG image."
            ),
        )


# ==========================================================
# WEATHER
# ==========================================================

@app.route(
    "/weather",
    methods=["GET", "POST"],
)
def weather():

    weather = None
    forecast = []
    weather_error = None

    if request.method == "POST":

        city = clean_text(
            request.form.get("city")
        )

        if not city:

            return render_template(
                "weather.html",
                weather=None,
                forecast=[],
                weather_error=(
                    "Please enter a city name."
                ),
            )

        if len(city) > 100:

            return render_template(
                "weather.html",
                weather=None,
                forecast=[],
                weather_error=(
                    "City name is too long."
                ),
            )

        try:

            # ------------------------------------------------
            # GEOCODING
            # ------------------------------------------------

            geo_url = (
                "https://geocoding-api.open-meteo.com/v1/search"
            )

            geo_params = {
                "name": city,
                "count": 1,
                "language": "en",
                "format": "json",
            }

            geo_response = requests.get(
                geo_url,
                params=geo_params,
                timeout=15,
            )

            geo_response.raise_for_status()

            geo_data = geo_response.json()

            if not geo_data.get("results"):

                return render_template(
                    "weather.html",
                    weather=None,
                    forecast=[],
                    weather_error=(
                        f"City '{city}' could not be found. "
                        "Please check the spelling and try again."
                    ),
                )

            location = geo_data[
                "results"
            ][0]

            latitude = location[
                "latitude"
            ]

            longitude = location[
                "longitude"
            ]

            city_name = location.get(
                "name",
                city,
            )

            country = location.get(
                "country",
                "",
            )

            # ------------------------------------------------
            # WEATHER API
            # ------------------------------------------------

            weather_url = (
                "https://api.open-meteo.com/v1/forecast"
            )

            weather_params = {

                "latitude": latitude,

                "longitude": longitude,

                "current": (
                    "temperature_2m,"
                    "relative_humidity_2m,"
                    "wind_speed_10m,"
                    "precipitation,"
                    "weather_code"
                ),

                "daily": (
                    "weather_code,"
                    "temperature_2m_max,"
                    "temperature_2m_min,"
                    "precipitation_sum,"
                    "precipitation_probability_max"
                ),

                "timezone": "auto",

                "forecast_days": 7,
            }

            weather_response = requests.get(
                weather_url,
                params=weather_params,
                timeout=15,
            )

            weather_response.raise_for_status()

            weather_data = (
                weather_response.json()
            )

            if weather_data.get("error"):

                raise RuntimeError(
                    weather_data.get(
                        "reason",
                        "Weather API returned an error.",
                    )
                )

            # ------------------------------------------------
            # CURRENT WEATHER
            # ------------------------------------------------

            current = weather_data.get(
                "current",
                {},
            )

            current_code = current.get(
                "weather_code"
            )

            weather = {

                "city": city_name,

                "country": country,

                "temperature": current.get(
                    "temperature_2m",
                    "-",
                ),

                "humidity": current.get(
                    "relative_humidity_2m",
                    "-",
                ),

                "rain": current.get(
                    "precipitation",
                    "-",
                ),

                "windspeed": current.get(
                    "wind_speed_10m",
                    "-",
                ),

                "weather_code": (
                    weather_code_to_text(
                        current_code
                    )
                ),

                "icon": (
                    weather_code_to_icon(
                        current_code
                    )
                ),
            }

            # ------------------------------------------------
            # DAILY FORECAST
            # ------------------------------------------------

            daily = weather_data.get(
                "daily"
            )

            if not daily:

                raise RuntimeError(
                    "Forecast data was not returned."
                )

            dates = daily.get(
                "time",
                [],
            )

            max_temps = daily.get(
                "temperature_2m_max",
                [],
            )

            min_temps = daily.get(
                "temperature_2m_min",
                [],
            )

            rainfall = daily.get(
                "precipitation_sum",
                [],
            )

            daily_codes = daily.get(
                "weather_code",
                [],
            )

            rain_probability = daily.get(
                "precipitation_probability_max",
                [],
            )

            total_days = min(
                len(dates),
                len(max_temps),
                len(min_temps),
                len(rainfall),
                len(daily_codes),
                len(rain_probability),
            )

            for i in range(total_days):

                date_object = datetime.strptime(
                    dates[i],
                    "%Y-%m-%d",
                )

                code = daily_codes[i]

                forecast.append({

                    "date": date_object.strftime(
                        "%d %b"
                    ),

                    "day": date_object.strftime(
                        "%A"
                    ),

                    "max_temp": round(
                        float(max_temps[i]),
                        1,
                    ),

                    "min_temp": round(
                        float(min_temps[i]),
                        1,
                    ),

                    "rainfall": round(
                        float(rainfall[i]),
                        1,
                    ),

                    "rain_probability": (
                        int(
                            rain_probability[i]
                        )
                        if rain_probability[i]
                        is not None
                        else 0
                    ),

                    "condition": (
                        weather_code_to_text(
                            code
                        )
                    ),

                    "icon": (
                        weather_code_to_icon(
                            code
                        )
                    ),
                })

            weather["forecast"] = forecast

            print(
                f"Weather loaded successfully "
                f"for {city_name}"
            )

            print(
                f"Forecast days received: "
                f"{len(forecast)}"
            )

        except requests.exceptions.Timeout:

            weather_error = (
                "The weather service took too long "
                "to respond. Please try again."
            )

        except requests.exceptions.RequestException as exc:

            print(
                "Weather API request error:",
                exc,
            )

            weather_error = (
                "Unable to connect to the weather service. "
                "Please check your internet connection "
                "and try again."
            )

        except Exception as exc:

            print(
                "Weather error:",
                exc,
            )

            weather_error = (
                "Unable to fetch weather information. "
                "Please try searching for the city again."
            )

        if weather_error:

            weather = None
            forecast = []

    return render_template(
        "weather.html",
        weather=weather,
        forecast=forecast,
        weather_error=weather_error,
    )


# ==========================================================
# MAHARASHTRA CROPS
# ==========================================================

@app.route("/maharashtra-crops")
def maharashtra_crops_page():

    return render_template(
        "maharashtra_crops.html",
        crops=maharashtra_crops,
    )


# ==========================================================
# CROP DETAILS
# ==========================================================

@app.route(
    "/crop/<crop_name>"
)
def crop_details(crop_name):

    crop = maharashtra_crops.get(
        crop_name
    )

    if crop is None:

        return render_template(
            "404.html",
            message="Crop not found.",
        ), 404

    return render_template(
        "crop_details.html",
        crop=crop,
    )


# ==========================================================
# HISTORY
# ==========================================================

@app.route("/history")
def history():

    history_data = []

    if HISTORY_FILE.exists():

        try:

            with HISTORY_FILE.open(
                "r",
                newline="",
                encoding="utf-8",
            ) as file:

                reader = csv.reader(
                    file
                )

                # Skip header
                next(
                    reader,
                    None,
                )

                for row in reader:

                    if row:
                        history_data.append(
                            row
                        )

        except Exception as exc:

            print(
                "History file error:",
                exc,
            )

    history_data.reverse()

    return render_template(
        "history.html",
        history=history_data,
    )


# ==========================================================
# CROP PREDICTION
# ==========================================================

@app.route(
    "/predict",
    methods=["POST"],
)
def predict():

    global latest_prediction
    global latest_details
    global latest_inputs
    global latest_prediction_time

    try:

        # Validate form
        inputs = validate_crop_inputs(
            request.form
        )

        # Prepare model input
        model_input = [[

            inputs["N"],

            inputs["P"],

            inputs["K"],

            inputs["temperature"],

            inputs["humidity"],

            inputs["ph"],

            inputs["rainfall"],

        ]]

        # Predict crop
        prediction = model.predict(
            model_input
        )

        result = clean_text(
            prediction[0],
            "Unknown",
        )

        # Crop information
        details = crop_info.get(
            result.lower(),
            {
                "fertilizer": "Not Available",
                "season": "Not Available",
                "water": "Not Available",
                "tips": (
                    "Information not available."
                ),
            },
        )

        # Save latest prediction
        latest_prediction = result

        latest_details = details

        latest_inputs = inputs

        latest_prediction_time = (
            datetime.now()
        )

        # Save history
        save_prediction_to_history(
            result,
            inputs,
        )

        return render_template(
            "result.html",

            prediction=result,

            details=details,

            moment=(
                latest_prediction_time.strftime(
                    "%d-%m-%Y %H:%M"
                )
            ),

            inputs=inputs,
        )

    except ValueError as exc:

        print(
            "Crop validation error:",
            exc,
        )

        return render_template(
            "predict.html",
            error=str(exc),
        )

    except Exception as exc:

        print(
            "Crop prediction error:",
            exc,
        )

        return render_template(
            "predict.html",
            error=(
                "Unable to predict the crop. "
                "Please check the entered values "
                "and try again."
            ),
        )


# ==========================================================
# DOWNLOAD PDF REPORT
# ==========================================================

@app.route("/download-report")
def download_report():

    if latest_prediction is None:

        return (
            "Please predict a crop first.",
            400,
        )

    try:

        ensure_directories()

        doc = SimpleDocTemplate(
            str(PDF_PATH),
            pagesize=A4,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40,
        )

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Title"],
            fontSize=26,
            leading=32,
            alignment=TA_CENTER,
            textColor=colors.HexColor(
                "#198754"
            ),
            spaceAfter=10,
        )

        subtitle_style = ParagraphStyle(
            "Subtitle",
            parent=styles["Heading2"],
            fontSize=16,
            leading=22,
            alignment=TA_CENTER,
            textColor=colors.HexColor(
                "#555555"
            ),
            spaceAfter=25,
        )

        heading_style = ParagraphStyle(
            "Heading",
            parent=styles["Heading2"],
            fontSize=15,
            leading=20,
            textColor=colors.HexColor(
                "#198754"
            ),
            spaceBefore=15,
            spaceAfter=10,
        )

        body_style = ParagraphStyle(
            "Body",
            parent=styles["BodyText"],
            fontSize=10.5,
            leading=16,
            textColor=colors.HexColor(
                "#333333"
            ),
            spaceAfter=8,
        )

        crop_style = ParagraphStyle(
            "Crop",
            parent=styles["Heading1"],
            fontSize=24,
            leading=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor(
                "#198754"
            ),
            spaceBefore=10,
            spaceAfter=15,
        )

        footer_style = ParagraphStyle(
            "Footer",
            parent=styles["Normal"],
            fontSize=9,
            alignment=TA_CENTER,
            textColor=colors.HexColor(
                "#777777"
            ),
        )

        story = []

        # --------------------------------------------------
        # HEADER
        # --------------------------------------------------

        story.append(
            Paragraph(
                "AgriVision AI",
                title_style,
            )
        )

        story.append(
            Paragraph(
                "Smart Agriculture Assistant",
                subtitle_style,
            )
        )

        story.append(
            Paragraph(
                "Crop Prediction Report",
                heading_style,
            )
        )

        story.append(
            Spacer(1, 10)
        )

        # --------------------------------------------------
        # RECOMMENDED CROP
        # --------------------------------------------------

        story.append(
            Paragraph(
                "AI Recommended Crop",
                heading_style,
            )
        )

        story.append(
            Paragraph(
                html.escape(
                    str(
                        latest_prediction
                    )
                ),
                crop_style,
            )
        )

        story.append(
            Paragraph(
                "This crop has been recommended "
                "by the AgriVision AI Machine "
                "Learning model based on the "
                "provided farming conditions.",
                body_style,
            )
        )

        story.append(
            Spacer(1, 10)
        )

        # --------------------------------------------------
        # INPUT VALUES
        # --------------------------------------------------

        if latest_inputs:

            story.append(
                Paragraph(
                    "Input Farming Conditions",
                    heading_style,
                )
            )

            input_rows = [

                [
                    Paragraph(
                        "<b>Parameter</b>",
                        body_style,
                    ),

                    Paragraph(
                        "<b>Value</b>",
                        body_style,
                    ),
                ],

                [
                    "Nitrogen (N)",
                    str(
                        latest_inputs["N"]
                    ),
                ],

                [
                    "Phosphorus (P)",
                    str(
                        latest_inputs["P"]
                    ),
                ],

                [
                    "Potassium (K)",
                    str(
                        latest_inputs["K"]
                    ),
                ],

                [
                    "Temperature (°C)",
                    str(
                        latest_inputs[
                            "temperature"
                        ]
                    ),
                ],

                [
                    "Humidity (%)",
                    str(
                        latest_inputs[
                            "humidity"
                        ]
                    ),
                ],

                [
                    "pH",
                    str(
                        latest_inputs["ph"]
                    ),
                ],

                [
                    "Rainfall (mm)",
                    str(
                        latest_inputs[
                            "rainfall"
                        ]
                    ),
                ],
            ]

            input_table = Table(
                input_rows,
                colWidths=[
                    250,
                    240,
                ],
            )

            input_table.setStyle(
                TableStyle([

                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor(
                            "#198754"
                        ),
                    ),

                    (
                        "TEXTCOLOR",
                        (0, 0),
                        (-1, 0),
                        colors.white,
                    ),

                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.HexColor(
                            "#cccccc"
                        ),
                    ),

                    (
                        "BACKGROUND",
                        (0, 1),
                        (-1, -1),
                        colors.HexColor(
                            "#f5fff8"
                        ),
                    ),

                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "MIDDLE",
                    ),

                    (
                        "LEFTPADDING",
                        (0, 0),
                        (-1, -1),
                        10,
                    ),

                    (
                        "RIGHTPADDING",
                        (0, 0),
                        (-1, -1),
                        10,
                    ),

                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        7,
                    ),

                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        7,
                    ),
                ])
            )

            story.append(
                input_table
            )

        # --------------------------------------------------
        # CROP INFORMATION
        # --------------------------------------------------

        story.append(
            Paragraph(
                "Crop Information",
                heading_style,
            )
        )

        details = (
            latest_details
            or {}
        )

        crop_data = [

            [
                Paragraph(
                    "<b>Information</b>",
                    body_style,
                ),

                Paragraph(
                    "<b>Recommendation</b>",
                    body_style,
                ),
            ],

            [
                "Recommended Fertilizer",
                clean_text(
                    details.get(
                        "fertilizer",
                        "Not Available",
                    ),
                    "Not Available",
                ),
            ],

            [
                "Best Season",
                clean_text(
                    details.get(
                        "season",
                        "Not Available",
                    ),
                    "Not Available",
                ),
            ],

            [
                "Water Requirement",
                clean_text(
                    details.get(
                        "water",
                        "Not Available",
                    ),
                    "Not Available",
                ),
            ],
        ]

        crop_table = Table(
            crop_data,
            colWidths=[
                210,
                280,
            ],
        )

        crop_table.setStyle(
            TableStyle([

                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#198754"
                    ),
                ),

                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor(
                        "#cccccc"
                    ),
                ),

                (
                    "BACKGROUND",
                    (0, 1),
                    (-1, -1),
                    colors.HexColor(
                        "#f5fff8"
                    ),
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),

                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    10,
                ),

                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    10,
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    10,
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    10,
                ),
            ])
        )

        story.append(
            crop_table
        )

        # --------------------------------------------------
        # FARMING TIP
        # --------------------------------------------------

        story.append(
            Paragraph(
                "Farming Tip",
                heading_style,
            )
        )

        story.append(
            Paragraph(
                html.escape(
                    clean_text(
                        details.get(
                            "tips",
                            "Information not available.",
                        ),
                        "Information not available.",
                    )
                ),
                body_style,
            )
        )

        # --------------------------------------------------
        # REPORT INFORMATION
        # --------------------------------------------------

        story.append(
            Spacer(1, 15)
        )

        story.append(
            Paragraph(
                "Report Information",
                heading_style,
            )
        )

        generated_time = (
            datetime.now().strftime(
                "%d-%m-%Y %H:%M"
            )
        )

        story.append(
            Paragraph(
                f"<b>Generated On:</b> "
                f"{generated_time}",
                body_style,
            )
        )

        story.append(
            Paragraph(
                "<b>System:</b> AgriVision AI "
                "Smart Agriculture Assistant",
                body_style,
            )
        )

        story.append(
            Paragraph(
                "<b>Purpose:</b> AI-based crop "
                "recommendation for smarter "
                "farming decisions.",
                body_style,
            )
        )

        story.append(
            Spacer(1, 25)
        )

        story.append(
            Paragraph(
                "© 2026 AgriVision AI | "
                "Diploma Final Year Project",
                footer_style,
            )
        )

        doc.build(
            story
        )

        return send_file(
            str(PDF_PATH),
            as_attachment=True,
            download_name=(
                "AgriVision_Crop_Prediction_Report.pdf"
            ),
            mimetype="application/pdf",
        )

    except Exception as exc:

        print(
            "PDF report error:",
            exc,
        )

        return (
            "Unable to generate the PDF report. "
            "Please try again.",
            500,
        )


# ==========================================================
# HEALTH CHECK
# ==========================================================

@app.route("/health")
def health():

    return {

        "status": "OK",

        "application": "AgriVision AI",

        "crop_model": (
            CROP_MODEL_PATH.exists()
        ),

        "leaf_model": (
            LEAF_MODEL_PATH.exists()
        ),

        "time": datetime.now().strftime(
            "%d-%m-%Y %H:%M:%S"
        ),
    }


# ==========================================================
# ERROR HANDLERS
# ==========================================================

@app.errorhandler(404)
def page_not_found(error):

    try:

        return render_template(
            "404.html",
            message=(
                "The page you are looking "
                "for was not found."
            ),
        ), 404

    except Exception:

        return (
            "404 - Page not found",
            404,
        )


@app.errorhandler(413)
def file_too_large(error):

    return render_template(
        "disease.html",
        error=(
            "Image is too large. "
            "Maximum allowed size is 5 MB."
        ),
    ), 413


@app.errorhandler(500)
def internal_server_error(error):

    print(
        "Internal server error:",
        error,
    )

    try:

        return render_template(
            "404.html",
            message=(
                "Something went wrong. "
                "Please try again."
            ),
        ), 500

    except Exception:

        return (
            "500 - Internal server error",
            500,
        )


# ==========================================================
# RUN APPLICATION
# ==========================================================

if __name__ == "__main__":

    ensure_directories()

    print("=" * 60)
    print(
        "🌾 AgriVision AI Smart Agriculture Assistant"
    )
    print("=" * 60)

    print(
        f"Crop model:"
    )
    print(
        f"  {CROP_MODEL_PATH}"
    )

    print(
        f"Leaf model:"
    )
    print(
        f"  {LEAF_MODEL_PATH}"
    )

    print(
        f"Crop model exists: "
        f"{CROP_MODEL_PATH.exists()}"
    )

    print(
        f"Leaf model exists: "
        f"{LEAF_MODEL_PATH.exists()}"
    )

    print(
        "Server starting..."
    )

    print("=" * 60)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )