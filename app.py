from flask import Flask, render_template, request, send_file
import joblib
import csv
import requests
import os
import random
import numpy as np

from PIL import Image
from datetime import datetime

from crop_info import crop_info
from maharashtra_crop import maharashtra_crops

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)

from reportlab.lib.enums import TA_CENTER


# ==========================================================
# FLASK APP
# ==========================================================

app = Flask(__name__)


# ==========================================================
# LOAD CROP MODEL
# ==========================================================

model = joblib.load(
    "model/crop_model.pkl"
)


# ==========================================================
# LOAD LEAF DISEASE MODEL
# ==========================================================

leaf_model = joblib.load(
    "model/leaf_disease_model.pkl"
)


# ==========================================================
# STORE LATEST CROP PREDICTION
# ==========================================================

latest_prediction = None
latest_details = None


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

    "📅 Keep a record of farming activities for better planning."

]


# ==========================================================
# HOME
# ==========================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        tip=random.choice(farming_tips)
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

@app.route("/contact")
def contact():

    return render_template(
        "contact.html"
    )


# ==========================================================
# LEAF DISEASE DETECTION
# ==========================================================

@app.route("/disease", methods=["GET", "POST"])
def disease():

    # ------------------------------------------------------
    # OPEN DISEASE PAGE
    # ------------------------------------------------------

    if request.method == "GET":

        return render_template(
            "disease.html"
        )


    # ------------------------------------------------------
    # CHECK IMAGE
    # ------------------------------------------------------

    if "leaf_image" not in request.files:

        return render_template(
            "disease.html",
            error="Please select a leaf image."
        )


    file = request.files["leaf_image"]


    # ------------------------------------------------------
    # CHECK EMPTY FILE
    # ------------------------------------------------------

    if file.filename == "":

        return render_template(
            "disease.html",
            error="Please select a leaf image."
        )


    try:

        # ==================================================
        # OPEN IMAGE
        # ==================================================

        image = Image.open(file)


        # ==================================================
        # CONVERT TO RGB
        # ==================================================

        image = image.convert("RGB")


        # ==================================================
        # RESIZE
        # ==================================================

        image = image.resize(
            (64, 64)
        )


        # ==================================================
        # CONVERT TO NUMPY
        # ==================================================

        image_array = np.array(image)


        # ==================================================
        # NORMALIZE
        # ==================================================

        image_array = image_array / 255.0


        # ==================================================
        # FLATTEN
        # ==================================================

        image_array = image_array.flatten()


        # ==================================================
        # MODEL INPUT
        # ==================================================

        image_array = image_array.reshape(
            1,
            -1
        )


        # ==================================================
        # PREDICTION
        # ==================================================

        prediction = leaf_model.predict(
            image_array
        )[0]


        # ==================================================
        # DISEASE INFORMATION
        # ==================================================

        disease_info = {

            # ------------------------------------------------
            # APPLE
            # ------------------------------------------------

            "Apple___Apple_scab": {
                "name": "Apple Scab",
                "description": "Apple scab is a fungal disease that causes dark lesions on apple leaves and fruit.",
                "advice": "Remove affected leaves and fallen plant material. Maintain good air circulation and avoid prolonged leaf wetness."
            },

            "Apple___Black_rot": {
                "name": "Apple Black Rot",
                "description": "Black rot can cause dark lesions on apple leaves and fruit.",
                "advice": "Remove infected plant parts and maintain good orchard sanitation."
            },

            "Apple___Cedar_apple_rust": {
                "name": "Apple Cedar Apple Rust",
                "description": "Cedar apple rust causes yellow-orange spots on apple leaves.",
                "advice": "Remove heavily affected leaves and improve air circulation around plants."
            },

            "Apple___healthy": {
                "name": "Healthy Apple Leaf",
                "description": "The uploaded apple leaf appears healthy.",
                "advice": "Continue regular monitoring, proper watering and balanced crop nutrition."
            },


            # ------------------------------------------------
            # BLUEBERRY
            # ------------------------------------------------

            "Blueberry___healthy": {
                "name": "Healthy Blueberry Leaf",
                "description": "The uploaded blueberry leaf appears healthy.",
                "advice": "Continue regular crop monitoring and maintain proper soil moisture."
            },


            # ------------------------------------------------
            # CHERRY
            # ------------------------------------------------

            "Cherry_(including_sour)___Powdery_mildew": {
                "name": "Cherry Powdery Mildew",
                "description": "Powdery mildew produces a white powder-like growth on leaves and shoots.",
                "advice": "Improve air circulation, avoid excessive humidity and remove severely affected plant parts."
            },

            "Cherry_(including_sour)___healthy": {
                "name": "Healthy Cherry Leaf",
                "description": "The uploaded cherry leaf appears healthy.",
                "advice": "Continue regular monitoring and maintain good orchard hygiene."
            },


            # ------------------------------------------------
            # CORN
            # ------------------------------------------------

            "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
                "name": "Corn Gray Leaf Spot",
                "description": "Gray leaf spot causes characteristic gray or brown lesions on corn leaves.",
                "advice": "Remove severely affected plant material and improve field airflow. Avoid prolonged leaf wetness."
            },

            "Corn_(maize)___Common_rust": {
                "name": "Corn Common Rust",
                "description": "Common rust produces reddish-brown rust-colored pustules on corn leaves.",
                "advice": "Monitor affected plants and maintain good crop management. Use resistant varieties where available."
            },

            "Corn_(maize)___Northern_Leaf_Blight": {
                "name": "Corn Northern Leaf Blight",
                "description": "Northern leaf blight produces elongated gray-green or brown lesions on corn leaves.",
                "advice": "Remove severely affected plant material and maintain good field sanitation."
            },

            "Corn_(maize)___healthy": {
                "name": "Healthy Corn Leaf",
                "description": "The uploaded corn leaf appears healthy.",
                "advice": "Continue regular crop monitoring and maintain appropriate irrigation and nutrition."
            },


            # ------------------------------------------------
            # GRAPE
            # ------------------------------------------------

            "Grape___Black_rot": {
                "name": "Grape Black Rot",
                "description": "Black rot causes brown lesions on grape leaves and can affect grape clusters.",
                "advice": "Remove infected plant material and improve air circulation around the vines."
            },

            "Grape___Esca_(Black_Measles)": {
                "name": "Grape Esca (Black Measles)",
                "description": "Esca is a grapevine disease that can cause characteristic leaf symptoms and fruit damage.",
                "advice": "Remove severely affected plant material and maintain vineyard sanitation."
            },

            "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
                "name": "Grape Leaf Blight",
                "description": "Grape leaf blight causes spots and lesions on grape leaves.",
                "advice": "Remove affected leaves and improve air circulation within the vineyard."
            },

            "Grape___healthy": {
                "name": "Healthy Grape Leaf",
                "description": "The uploaded grape leaf appears healthy.",
                "advice": "Continue regular vineyard monitoring and proper crop care."
            },


            # ------------------------------------------------
            # ORANGE
            # ------------------------------------------------

            "Orange___Haunglongbing_(Citrus_greening)": {
                "name": "Citrus Greening",
                "description": "Citrus greening can cause yellowing and abnormal leaf development in citrus plants.",
                "advice": "Monitor affected plants carefully and consult an agricultural expert for appropriate management."
            },


            # ------------------------------------------------
            # PEACH
            # ------------------------------------------------

            "Peach___Bacterial_spot": {
                "name": "Peach Bacterial Spot",
                "description": "Bacterial spot can produce dark spots and lesions on peach leaves.",
                "advice": "Remove severely affected plant parts and maintain good orchard sanitation."
            },

            "Peach___healthy": {
                "name": "Healthy Peach Leaf",
                "description": "The uploaded peach leaf appears healthy.",
                "advice": "Continue regular monitoring and proper orchard care."
            },


            # ------------------------------------------------
            # PEPPER
            # ------------------------------------------------

            "Pepper,_bell___Bacterial_spot": {
                "name": "Pepper Bacterial Spot",
                "description": "Bacterial spot causes dark lesions on pepper leaves and fruit.",
                "advice": "Remove affected plant material and avoid overhead watering."
            },

            "Pepper,_bell___healthy": {
                "name": "Healthy Pepper Leaf",
                "description": "The uploaded pepper leaf appears healthy.",
                "advice": "Continue regular monitoring and proper watering."
            },


            # ------------------------------------------------
            # POTATO
            # ------------------------------------------------

            "Potato___Early_blight": {
                "name": "Potato Early Blight",
                "description": "Early blight causes dark lesions and concentric ring patterns on potato leaves.",
                "advice": "Remove affected leaves, maintain good field sanitation and avoid prolonged leaf wetness."
            },

            "Potato___Late_blight": {
                "name": "Potato Late Blight",
                "description": "Late blight can cause dark, water-soaked lesions on potato leaves.",
                "advice": "Remove infected plant material, improve air circulation and avoid overhead watering."
            },

            "Potato___healthy": {
                "name": "Healthy Potato Leaf",
                "description": "The uploaded potato leaf appears healthy.",
                "advice": "Continue regular crop monitoring and proper crop care."
            },


            # ------------------------------------------------
            # RASPBERRY
            # ------------------------------------------------

            "Raspberry___healthy": {
                "name": "Healthy Raspberry Leaf",
                "description": "The uploaded raspberry leaf appears healthy.",
                "advice": "Continue regular monitoring and maintain appropriate watering and nutrition."
            },


            # ------------------------------------------------
            # SOYBEAN
            # ------------------------------------------------

            "Soybean___healthy": {
                "name": "Healthy Soybean Leaf",
                "description": "The uploaded soybean leaf appears healthy.",
                "advice": "Continue regular monitoring and proper crop management."
            },


            # ------------------------------------------------
            # SQUASH
            # ------------------------------------------------

            "Squash___Powdery_mildew": {
                "name": "Squash Powdery Mildew",
                "description": "Powdery mildew produces a white powder-like growth on squash leaves.",
                "advice": "Improve air circulation and remove severely affected leaves."
            },


            # ------------------------------------------------
            # STRAWBERRY
            # ------------------------------------------------

            "Strawberry___Leaf_scorch": {
                "name": "Strawberry Leaf Scorch",
                "description": "Leaf scorch produces dark lesions and scorched areas on strawberry leaves.",
                "advice": "Remove severely affected leaves and maintain proper irrigation and field sanitation."
            },

            "Strawberry___healthy": {
                "name": "Healthy Strawberry Leaf",
                "description": "The uploaded strawberry leaf appears healthy.",
                "advice": "Continue regular monitoring and proper crop care."
            },


            # ------------------------------------------------
            # TOMATO
            # ------------------------------------------------

            "Tomato___Bacterial_spot": {
                "name": "Tomato Bacterial Spot",
                "description": "Bacterial spot causes small dark lesions on tomato leaves and fruit.",
                "advice": "Remove affected leaves and avoid overhead watering."
            },

            "Tomato___Early_blight": {
                "name": "Tomato Early Blight",
                "description": "Early blight causes dark lesions with characteristic ring patterns on tomato leaves.",
                "advice": "Remove affected leaves, improve air circulation and avoid excess moisture."
            },

            "Tomato___Late_blight": {
                "name": "Tomato Late Blight",
                "description": "Late blight can cause dark brown or black lesions on tomato leaves and other plant parts.",
                "advice": "Remove infected leaves and severely affected plant parts. Improve air circulation and avoid overhead watering."
            },

            "Tomato___Leaf_Mold": {
                "name": "Tomato Leaf Mold",
                "description": "Leaf mold can cause yellow areas on the upper surface of tomato leaves and fungal growth underneath.",
                "advice": "Improve ventilation, reduce humidity and avoid prolonged leaf wetness."
            },

            "Tomato___Septoria_leaf_spot": {
                "name": "Tomato Septoria Leaf Spot",
                "description": "Septoria leaf spot produces numerous small spots on tomato leaves.",
                "advice": "Remove affected leaves, maintain field sanitation and avoid overhead watering."
            },

            "Tomato___Spider_mites Two-spotted_spider_mite": {
                "name": "Tomato Spider Mites",
                "description": "Spider mites can cause speckling, yellowing and damage to tomato leaves.",
                "advice": "Inspect the underside of leaves and manage the infestation using appropriate agricultural pest-control practices."
            },

            "Tomato___Target_Spot": {
                "name": "Tomato Target Spot",
                "description": "Target spot produces circular lesions on tomato leaves.",
                "advice": "Remove severely affected leaves and improve air circulation."
            },

            "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
                "name": "Tomato Yellow Leaf Curl Virus",
                "description": "This viral disease can cause leaf curling, yellowing and stunted plant growth.",
                "advice": "Remove severely infected plants and manage insect vectors such as whiteflies with appropriate agricultural practices."
            },

            "Tomato___Tomato_mosaic_virus": {
                "name": "Tomato Mosaic Virus",
                "description": "Tomato mosaic virus can cause mottled leaf patterns and reduced plant growth.",
                "advice": "Remove severely affected plants and maintain good tool and field sanitation."
            },

            "Tomato___healthy": {
                "name": "Healthy Tomato Leaf",
                "description": "The uploaded tomato leaf appears healthy.",
                "advice": "Continue regular monitoring, proper watering and balanced crop nutrition."
            }
        }


        # ==================================================
        # GET DISEASE INFORMATION
        # ==================================================

        info = disease_info.get(
            prediction
        )


        # ==================================================
        # FALLBACK
        # ==================================================

        if info is None:

            info = {

                "name": prediction.replace(
                    "___",
                    " - "
                ),

                "description":
                    "The AI model detected this PlantVillage class.",

                "advice":
                    "Monitor the crop regularly and consult an agricultural expert if symptoms continue."

            }


        # ==================================================
        # RESULT PAGE
        # ==================================================

        return render_template(

            "disease_result.html",

            prediction=info["name"],

            recommendation=info["advice"],

            description=info["description"]

        )


    except Exception as e:

        print(
            "Leaf prediction error:",
            e
        )

        return render_template(

            "disease.html",

            error=(
                "Unable to analyze the image. "
                "Please upload a valid JPG or PNG image."
            )

        )


# ==========================================================
# WEATHER
# ==========================================================

@app.route("/weather", methods=["GET", "POST"])
def weather():

    weather = None

    if request.method == "POST":

        city = request.form["city"]

        try:

            # ------------------------------------------------
            # GEOCODING
            # ------------------------------------------------

            geo_url = (
                "https://geocoding-api.open-meteo.com/v1/search"
                f"?name={city}&count=1"
            )

            geo_response = requests.get(
                geo_url
            ).json()


            if "results" in geo_response:

                latitude = (
                    geo_response["results"][0]["latitude"]
                )

                longitude = (
                    geo_response["results"][0]["longitude"]
                )

                city_name = (
                    geo_response["results"][0]["name"]
                )


                # ------------------------------------------------
                # WEATHER API
                # ------------------------------------------------

                weather_url = (

                    "https://api.open-meteo.com/v1/forecast?"

                    f"latitude={latitude}"

                    f"&longitude={longitude}"

                    "&current="

                    "temperature_2m,"

                    "relative_humidity_2m,"

                    "wind_speed_10m,"

                    "precipitation,"

                    "weather_code"

                )


                weather_response = requests.get(
                    weather_url
                ).json()


                current = (
                    weather_response["current"]
                )


                # ------------------------------------------------
                # WEATHER CODES
                # ------------------------------------------------

                weather_codes = {

                    0: "☀️ Clear Sky",
                    1: "🌤 Mainly Clear",
                    2: "⛅ Partly Cloudy",
                    3: "☁️ Cloudy",

                    45: "🌫 Fog",
                    48: "🌫 Dense Fog",

                    51: "🌦 Light Drizzle",
                    53: "🌦 Moderate Drizzle",
                    55: "🌧 Heavy Drizzle",

                    61: "🌦 Light Rain",
                    63: "🌧 Moderate Rain",
                    65: "🌧 Heavy Rain",

                    71: "❄️ Light Snow",
                    73: "❄️ Moderate Snow",
                    75: "❄️ Heavy Snow",

                    80: "🌦 Rain Showers",
                    81: "🌧 Heavy Showers",
                    82: "🌧 Violent Showers",

                    95: "⛈ Thunderstorm",
                    96: "⛈ Thunderstorm with Hail",
                    99: "⛈ Severe Thunderstorm"

                }


                # ------------------------------------------------
                # WEATHER ICONS
                # ------------------------------------------------

                weather_icons = {

                    0: "☀️",
                    1: "🌤",
                    2: "⛅",
                    3: "☁️",

                    45: "🌫",
                    48: "🌫",

                    51: "🌦",
                    53: "🌦",
                    55: "🌧",

                    61: "🌦",
                    63: "🌧",
                    65: "🌧",

                    71: "❄️",
                    73: "❄️",
                    75: "❄️",

                    80: "🌦",
                    81: "🌧",
                    82: "🌧",

                    95: "⛈",
                    96: "⛈",
                    99: "⛈"

                }


                # ------------------------------------------------
                # WEATHER DATA
                # ------------------------------------------------

                weather = {

                    "city":
                        city_name,

                    "temperature":
                        current["temperature_2m"],

                    "humidity":
                        current["relative_humidity_2m"],

                    "rain":
                        current["precipitation"],

                    "windspeed":
                        current["wind_speed_10m"],

                    "weather_code":
                        weather_codes.get(
                            current["weather_code"],
                            "Unknown Weather"
                        ),

                    "icon":
                        weather_icons.get(
                            current["weather_code"],
                            "🌍"
                        )

                }


            else:

                weather = {

                    "city": city,

                    "temperature": "Not Found",

                    "humidity": "-",

                    "windspeed": "-",

                    "rain": "-",

                    "weather_code": "-"

                }


        except Exception as e:

            print(
                "Weather error:",
                e
            )

            weather = {

                "city": city,

                "temperature": "Error",

                "humidity": "-",

                "windspeed": "-",

                "rain": "-",

                "weather_code": "-"

            }


    return render_template(

        "weather.html",

        weather=weather

    )


# ==========================================================
# MAHARASHTRA CROPS
# ==========================================================

@app.route("/maharashtra-crops")
def maharashtra_crops_page():

    return render_template(

        "maharashtra_crops.html",

        crops=maharashtra_crops

    )


# ==========================================================
# CROP DETAILS
# ==========================================================

@app.route("/crop/<crop_name>")
def crop_details(crop_name):

    crop = maharashtra_crops.get(
        crop_name
    )


    if crop is None:

        return "Crop not found!"


    return render_template(

        "crop_details.html",

        crop=crop

    )


# ==========================================================
# HISTORY
# ==========================================================

@app.route("/history")
def history():

    history_data = []

    history_file = (
        "saved_predictions/prediction_history.csv"
    )


    if os.path.exists(history_file):

        with open(
            history_file,
            "r",
            newline=""
        ) as file:

            reader = csv.reader(
                file
            )

            next(
                reader,
                None
            )


            for row in reader:

                history_data.append(
                    row
                )


    return render_template(

        "history.html",

        history=history_data

    )


# ==========================================================
# CROP PREDICTION
# ==========================================================

@app.route("/predict", methods=["POST"])
def predict():

    global latest_prediction
    global latest_details


    # ------------------------------------------------------
    # INPUTS
    # ------------------------------------------------------

    N = float(
        request.form["N"]
    )

    P = float(
        request.form["P"]
    )

    K = float(
        request.form["K"]
    )

    temperature = float(
        request.form["temperature"]
    )

    humidity = float(
        request.form["humidity"]
    )

    ph = float(
        request.form["ph"]
    )

    rainfall = float(
        request.form["rainfall"]
    )


    # ------------------------------------------------------
    # CROP MODEL PREDICTION
    # ------------------------------------------------------

    prediction = model.predict([

        [
            N,
            P,
            K,
            temperature,
            humidity,
            ph,
            rainfall
        ]

    ])


    result = prediction[0]


    # ------------------------------------------------------
    # CROP DETAILS
    # ------------------------------------------------------

    details = crop_info.get(

        result.lower(),

        {

            "fertilizer":
                "Not Available",

            "season":
                "Not Available",

            "water":
                "Not Available",

            "tips":
                "Information not available."

        }

    )


    # ------------------------------------------------------
    # STORE LATEST RESULT
    # ------------------------------------------------------

    latest_prediction = result

    latest_details = details


    # ------------------------------------------------------
    # CREATE HISTORY FOLDER
    # ------------------------------------------------------

    os.makedirs(
        "saved_predictions",
        exist_ok=True
    )


    history_file = (
        "saved_predictions/prediction_history.csv"
    )


    # ------------------------------------------------------
    # CREATE CSV
    # ------------------------------------------------------

    if not os.path.exists(history_file):

        with open(
            history_file,
            "w",
            newline=""
        ) as file:

            writer = csv.writer(
                file
            )

            writer.writerow([

                "Date",

                "Crop",

                "Nitrogen",

                "Phosphorus",

                "Potassium",

                "Temperature",

                "Humidity",

                "pH",

                "Rainfall"

            ])


    # ------------------------------------------------------
    # SAVE PREDICTION
    # ------------------------------------------------------

    with open(
        history_file,
        "a",
        newline=""
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([

            datetime.now().strftime(
                "%d-%m-%Y %H:%M"
            ),

            result,

            N,

            P,

            K,

            temperature,

            humidity,

            ph,

            rainfall

        ])


    # ------------------------------------------------------
    # RESULT
    # ------------------------------------------------------

    return render_template(

        "result.html",

        prediction=result,

        details=details,

        moment=datetime.now().strftime(
            "%d-%m-%Y %H:%M"
        )

    )


# ==========================================================
# DOWNLOAD PDF REPORT
# ==========================================================

@app.route("/download-report")
def download_report():

    global latest_prediction
    global latest_details


    # ------------------------------------------------------
    # CHECK PREDICTION
    # ------------------------------------------------------

    if latest_prediction is None:

        return (
            "Please predict a crop first."
        )


    # ------------------------------------------------------
    # PDF PATH
    # ------------------------------------------------------

    pdf_path = (
        "prediction_report.pdf"
    )


    # ------------------------------------------------------
    # CREATE PDF
    # ------------------------------------------------------

    doc = SimpleDocTemplate(

        pdf_path,

        pagesize=A4,

        rightMargin=40,

        leftMargin=40,

        topMargin=40,

        bottomMargin=40

    )


    # ------------------------------------------------------
    # STYLES
    # ------------------------------------------------------

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

        spaceAfter=10

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

        spaceAfter=25

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

        spaceAfter=10

    )


    body_style = ParagraphStyle(

        "Body",

        parent=styles["BodyText"],

        fontSize=11,

        leading=17,

        textColor=colors.HexColor(
            "#333333"
        ),

        spaceAfter=8

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

        spaceAfter=15

    )


    footer_style = ParagraphStyle(

        "Footer",

        parent=styles["Normal"],

        fontSize=9,

        alignment=TA_CENTER,

        textColor=colors.HexColor(
            "#777777"
        )

    )


    story = []


    # ------------------------------------------------------
    # HEADER
    # ------------------------------------------------------

    story.append(

        Paragraph(

            "AgriVision AI",

            title_style

        )

    )


    story.append(

        Paragraph(

            "Smart Agriculture Assistant",

            subtitle_style

        )

    )


    story.append(

        Paragraph(

            "Crop Prediction Report",

            heading_style

        )

    )


    story.append(
        Spacer(1, 10)
    )


    # ------------------------------------------------------
    # PREDICTED CROP
    # ------------------------------------------------------

    story.append(

        Paragraph(

            "AI Recommended Crop",

            heading_style

        )

    )


    story.append(

        Paragraph(

            f"{latest_prediction}",

            crop_style

        )

    )


    story.append(

        Paragraph(

            "This crop has been recommended by the "
            "AgriVision AI Machine Learning model "
            "based on the provided farming conditions.",

            body_style

        )

    )


    story.append(
        Spacer(1, 10)
    )


    # ------------------------------------------------------
    # CROP INFORMATION TABLE
    # ------------------------------------------------------

    crop_data = [

        [

            Paragraph(
                "<b>Information</b>",
                body_style
            ),

            Paragraph(
                "<b>Recommendation</b>",
                body_style
            )

        ],

        [

            "Recommended Fertilizer",

            latest_details.get(
                "fertilizer",
                "Not Available"
            )

        ],

        [

            "Best Season",

            latest_details.get(
                "season",
                "Not Available"
            )

        ],

        [

            "Water Requirement",

            latest_details.get(
                "water",
                "Not Available"
            )

        ]

    ]


    crop_table = Table(

        crop_data,

        colWidths=[
            210,
            280
        ]

    )


    crop_table.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#198754")
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#cccccc")
            ),

            (
                "BACKGROUND",
                (0, 1),
                (-1, -1),
                colors.HexColor("#f5fff8")
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),

            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                10
            ),

            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                10
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                10
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                10
            )

        ])

    )


    story.append(
        crop_table
    )


    # ------------------------------------------------------
    # FARMING TIP
    # ------------------------------------------------------

    story.append(

        Paragraph(

            "Farming Tip",

            heading_style

        )

    )


    story.append(

        Paragraph(

            latest_details.get(

                "tips",

                "Information not available."

            ),

            body_style

        )

    )


    # ------------------------------------------------------
    # REPORT INFORMATION
    # ------------------------------------------------------

    story.append(
        Spacer(1, 15)
    )


    story.append(

        Paragraph(

            "Report Information",

            heading_style

        )

    )


    story.append(

        Paragraph(

            f"<b>Generated On:</b> "
            f"{datetime.now().strftime('%d-%m-%Y %H:%M')}",

            body_style

        )

    )


    story.append(

        Paragraph(

            "<b>System:</b> "
            "AgriVision AI Smart Agriculture Assistant",

            body_style

        )

    )


    story.append(

        Paragraph(

            "<b>Purpose:</b> "
            "AI-based crop recommendation "
            "for smarter farming decisions.",

            body_style

        )

    )


    # ------------------------------------------------------
    # FOOTER
    # ------------------------------------------------------

    story.append(
        Spacer(1, 25)
    )


    story.append(

        Paragraph(

            "© 2026 AgriVision AI | "
            "Diploma Final Year Project",

            footer_style

        )

    )


    # ------------------------------------------------------
    # BUILD PDF
    # ------------------------------------------------------

    doc.build(
        story
    )


    return send_file(

        pdf_path,

        as_attachment=True

    )


# ==========================================================
# RUN APPLICATION
# ==========================================================

if __name__ == "__main__":

    app.run(
        debug=False
    )