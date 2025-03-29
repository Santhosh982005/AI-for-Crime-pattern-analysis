# Set matplotlib backend first to avoid tkinter issues
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

import os
import folium
import tensorflow as tf
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template, send_from_directory
import requests
import pandas as pd
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import json

# Initialize Flask app
app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

# Configuration
app.config['MODEL_DIR'] = "models"
app.config['PLOT_DIR'] = "static/plots"
app.config['MAP_DIR'] = "static/maps"
app.config['ALLOWED_EXTENSIONS'] = {'csv'}
app.config['QA_PAIRS_PATH'] = "data/trained_qa.json"

# Ensure directories exist
os.makedirs(app.config['MODEL_DIR'], exist_ok=True)
os.makedirs(app.config['PLOT_DIR'], exist_ok=True)
os.makedirs(app.config['MAP_DIR'], exist_ok=True)
os.makedirs("data", exist_ok=True)

# Load environment variables
load_dotenv()

# Load models at startup
def load_models():
    """Load all ML models at application startup"""
    models = {}
    try:
        # Load trend prediction model
        models['trend'] = tf.keras.models.load_model(
            os.path.join(app.config['MODEL_DIR'], "crime_trend_model.h5"),
            compile=False
        )
        models['trend'].compile(optimizer='adam', loss='mse')
        models['scaler'] = joblib.load(
            os.path.join(app.config['MODEL_DIR'], "scaler.pkl")
        )
        
        # Load hotspot model
        models['hotspot'] = joblib.load(
            os.path.join(app.config['MODEL_DIR'], "crime_hotspot_model.pkl")
        )
        
        # Load crime type prediction model
        models['crime'] = joblib.load(
            os.path.join(app.config['MODEL_DIR'], "crime_prediction_model.pkl")
        )
        models['encoder'] = joblib.load(
            os.path.join(app.config['MODEL_DIR'], "encoder.pkl")
        )
        
        print("✅ All models loaded successfully!")
        return models
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        return None

# Load models
models = load_models()

# Initialize simple Q&A database
def init_qa_pairs():
    if not os.path.exists(app.config['QA_PAIRS_PATH']):
        sample_qa = [
            {"question": "what crimes happen at night", "answer": "Common night crimes (8PM-4AM) include burglary (35%), vandalism (25%), and DUI (20%)."},
            {"question": "where do thefts occur", "answer": "Theft hotspots: Downtown (30%), Shopping Districts (25%), and Transit Areas (20%)."},
            {"question": "common weekend crimes", "answer": "Weekends see more assaults (40%), theft (30%), and vandalism (25%)."},
            {"question": "crime prevention tips", "answer": "Top prevention tips: 1) Install security cameras 2) Improve lighting 3) Report suspicious activity"},
            {"question": "latest crime trends", "answer": "Current trends show a 15% decrease in burglaries but 20% increase in online fraud"}
        ]
        with open(app.config['QA_PAIRS_PATH'], 'w') as f:
            json.dump(sample_qa, f)

init_qa_pairs()

def generate_trend_plot(history, predictions, future_dates):
    """Generate a plot showing historical and predicted crime trends"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        plot_filename = f"crime_trend_{timestamp}.png"
        plot_path = os.path.join(app.config['PLOT_DIR'], plot_filename)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical data
        ax.plot(history.index, history.values, 
                label="Historical Crime Data", color='blue')
        
        # Plot predictions
        ax.plot(future_dates, predictions,
                label="Predicted Crime Trend", color='red', linestyle='--')
        
        # Formatting
        ax.set_title("Crime Trend Prediction")
        ax.set_xlabel("Date")
        ax.set_ylabel("Crime Count")
        ax.legend()
        ax.grid(True)
        
        # Format x-axis dates
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.autofmt_xdate()
        
        plt.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)
        
        return plot_filename
    except Exception as e:
        print(f"Error generating plot: {e}")
        raise

def generate_hotspot_map(hotspots, center_location):
    """Generate Folium map with crime hotspots"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    map_filename = f"crime_hotspots_{timestamp}.html"
    map_path = os.path.join(app.config['MAP_DIR'], map_filename)
    
    # Create map with proper bounds checking
    crime_map = folium.Map(
        location=center_location,
        zoom_start=12,
        control_scale=True
    )
    
    # Add tile layer
    folium.TileLayer('openstreetmap').add_to(crime_map)
    
    # Filter out invalid coordinates
    valid_hotspots = [
        loc for loc in hotspots 
        if abs(loc[0]) <= 90 and abs(loc[1]) <= 180  # Valid lat/long ranges
    ]
    
    if not valid_hotspots:
        valid_hotspots = [center_location]  # Fallback to center if no valid points
    
    # Add markers for each valid hotspot
    for i, location in enumerate(valid_hotspots):
        folium.CircleMarker(
            location=location,
            radius=10,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.7,
            popup=f"Hotspot {i+1}",
            tooltip=f"Lat: {location[0]:.4f}, Long: {location[1]:.4f}"
        ).add_to(crime_map)
    
    # Fit map to bounds of all markers
    if len(valid_hotspots) > 1:
        crime_map.fit_bounds([
            [min(loc[0] for loc in valid_hotspots), min(loc[1] for loc in valid_hotspots)],
            [max(loc[0] for loc in valid_hotspots), max(loc[1] for loc in valid_hotspots)]
        ])
    
    crime_map.save(map_path)
    return map_filename

@app.route("/")
def home():
    """Render the main dashboard page"""
    return render_template("index.html")

@app.route("/predict-trend", methods=["GET"])
def predict_trend():
    """Predict crime trends using LSTM model"""
    if not models or 'trend' not in models:
        return render_template("error.html", error_message="Trend prediction model not loaded")
    
    try:
        # Load sample historical data (in production, use real data)
        dataset_path = "dataset/Crime_Data_from_2020_to_Present.csv"
        if not os.path.exists(dataset_path):
            return render_template("error.html", error_message="Dataset not found")
        
        # Get last 90 days of data
        crime_data = pd.read_csv(dataset_path)
        crime_data['DATE OCC'] = pd.to_datetime(crime_data['DATE OCC'])
        history = crime_data.groupby('DATE OCC').size().sort_index()
        history = history.last('90D')  # Last 90 days
        
        if len(history) < 30:
            return render_template("error.html", error_message="Insufficient historical data")
        
        # Prepare data for prediction
        scaled_data = models['scaler'].transform(history.values.reshape(-1, 1)).flatten()
        
        # Make predictions for next 30 days
        n_steps = 90
        predictions = []
        current_batch = scaled_data[-n_steps:].reshape(1, n_steps, 1)
        
        for _ in range(30):  # Predict next 30 days
            next_pred = models['trend'].predict(current_batch)[0]
            predictions.append(next_pred[0])
            current_batch = np.append(current_batch[:, 1:, :], [[next_pred]], axis=1)
        
        # Inverse transform predictions
        predictions = models['scaler'].inverse_transform(
            np.array(predictions).reshape(-1, 1)).flatten()
        
        # Generate dates for plotting
        last_date = history.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
        
        # Generate plot
        plot_filename = generate_trend_plot(history, predictions, future_dates)
        
        return render_template("results.html",
                            prediction_type="Crime Trend",
                            plot_url=f"/static/plots/{plot_filename}",
                            additional_data="30-day forecast")
    
    except Exception as e:
        return render_template("error.html", error_message=str(e))

@app.route("/predict-hotspots", methods=["GET"])
def predict_hotspots():
    """Predict crime hotspots using K-Means model"""
    if not models or 'hotspot' not in models:
        return render_template("error.html", error_message="Hotspot detection model not loaded")
    
    try:
        # Get hotspot locations
        hotspots = models['hotspot'].cluster_centers_
        
        # Calculate map center
        center_location = [hotspots[:, 0].mean(), hotspots[:, 1].mean()]
        
        # Generate map
        map_filename = generate_hotspot_map(hotspots, center_location)
        
        return render_template("results.html",
                            prediction_type="Crime Hotspots",
                            map_url=f"/static/maps/{map_filename}",
                            additional_data=f"{len(hotspots)} high-risk areas identified")
    
    except Exception as e:
        return render_template("error.html", error_message=str(e))

@app.route("/predict-crime", methods=["GET", "POST"])
def predict_crime():
    """Predict crime type based on location and time"""
    if not models or 'crime' not in models or 'encoder' not in models:
        return render_template("error.html", error_message="Crime prediction model not loaded")
    
    if request.method == "GET":
        return render_template("crime_form.html")
    
    try:
        # Get form data
        lat = float(request.form.get("lat"))
        lon = float(request.form.get("lon"))
        hour = int(request.form.get("hour", 12))  # Default to noon
        
        # Make prediction
        pred = models['crime'].predict([[lat, lon, hour]])
        crime_type = models['encoder'].inverse_transform(pred)
        
        return render_template("crime_result.html",
                            crime_type=crime_type[0],
                            location=f"Latitude: {lat}, Longitude: {lon}",
                            time=f"{hour}:00")
    
    except Exception as e:
        return render_template("error.html", error_message=str(e))

@app.route("/news", methods=["GET"])
def get_crime_news():
    """Fetch crime-related news with proper API key handling"""
    try:
        # Get API key from environment variables
        API_KEY = os.getenv("NEWS_API_KEY", "339ff28e97fa4c06966466d1bb6261c7")
        
        if not API_KEY:
            return render_template("error.html",
                                error_message="News API key not configured. Please contact administrator.")

        # API request parameters
        params = {
            "q": "crime",
            "apiKey": API_KEY,
            "pageSize": 5,
            "sortBy": "publishedAt",
            "language": "en",
            "domains": "bbc.co.uk,cnn.com,nytimes.com"
        }

        # Make the API request
        response = requests.get(
            "https://newsapi.org/v2/everything",
            params=params,
            timeout=10
        )

        # Check for API errors
        if response.status_code != 200:
            error_msg = response.json().get('message', 'Unknown error')
            return render_template("error.html",
                                error_message=f"News API Error: {error_msg}")

        news_data = response.json()
        articles = news_data.get('articles', [])

        # Process articles for display
        processed_articles = []
        for article in articles:
            processed_articles.append({
                'title': article.get('title', 'No title available'),
                'description': article.get('description', 'No description available'),
                'url': article.get('url', '#'),
                'source': article.get('source', {}).get('name', 'Unknown'),
                'date': article.get('publishedAt', '')[:10],  # Just the date part
                'image': article.get('urlToImage', '')
            })

        return render_template("news.html",
                            articles=processed_articles,
                            total_results=news_data.get('totalResults', 0))

    except requests.exceptions.RequestException as e:
        return render_template("error.html",
                            error_message=f"Failed to fetch news: {str(e)}")

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    """Simple crime data chatbot"""
    # Load Q&A pairs
    with open(app.config['QA_PAIRS_PATH']) as f:
        qa_pairs = json.load(f)
    
    answer = None
    question = ""
    
    if request.method == 'POST':
        question = request.form.get('question', '').lower()
        
        # Simple keyword matching
        best_match = None
        best_score = 0
        
        for pair in qa_pairs:
            q_words = pair['question'].split()
            score = sum(1 for word in q_words if word in question)
            
            if score > best_score:
                best_score = score
                best_match = pair['answer']
        
        answer = best_match if best_score > 1 else "I can answer about crime patterns. Try asking about specific crimes, times, or locations."

    return render_template('chatbot.html', 
                         question=question,
                         answer=answer,
                         show_result=request.method == 'POST')

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)