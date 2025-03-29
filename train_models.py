import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(filepath):
    """Load and preprocess crime data"""
    df = pd.read_csv(filepath)
    
    # Convert and clean date column
    df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], errors='coerce')
    df = df.dropna(subset=['DATE OCC'])
    
    # Aggregate crimes by date
    df_trend = df.groupby('DATE OCC').size().reset_index(name='Crime_Count')
    df_trend.set_index('DATE OCC', inplace=True)
    
    # Fill missing dates with 0 crimes
    date_range = pd.date_range(start=df_trend.index.min(), end=df_trend.index.max())
    df_trend = df_trend.reindex(date_range, fill_value=0)
    
    return df_trend['Crime_Count']

def create_sequences(data, n_steps):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i-n_steps:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def train_lstm_model(data, model_dir, n_steps=90, epochs=50, batch_size=32):
    """Train LSTM model for crime trend prediction"""
    # Normalize data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1)).flatten()
    
    # Create sequences
    X, y = create_sequences(scaled_data, n_steps)
    
    # Split into train/test
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Build LSTM model
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(n_steps, 1)),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    # Train with early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    history = model.fit(
        X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test.reshape(X_test.shape[0], X_test.shape[1], 1), y_test),
        callbacks=[early_stop],
        verbose=1
    )
    
    # Save model and scaler
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, 'crime_trend_model.h5'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    
    return model, history

def train_hotspot_model(df, model_dir):
    """Train KMeans model for crime hotspots"""
    if 'LAT' in df.columns and 'LON' in df.columns:
        df_hotspot = df[['LAT', 'LON']].dropna()
        
        # Train K-Means model
        kmeans = KMeans(n_clusters=10, random_state=42, n_init=20)
        kmeans.fit(df_hotspot)
        
        # Save model
        joblib.dump(kmeans, os.path.join(model_dir, 'crime_hotspot_model.pkl'))
        return kmeans
    return None

def train_crime_type_model(df, model_dir):
    """Train RandomForest model for crime type prediction"""
    required_cols = ['LAT', 'LON', 'TIME OCC', 'Crm Cd Desc']
    if all(col in df.columns for col in required_cols):
        df_pred = df[required_cols].dropna().copy()
        df_pred.columns = ['lat', 'lon', 'hour', 'crime_type']
        
        # Encode crime types
        encoder = LabelEncoder()
        df_pred['crime_encoded'] = encoder.fit_transform(df_pred['crime_type'])
        
        # Train RandomForest
        rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
        rf.fit(df_pred[['lat', 'lon', 'hour']], df_pred['crime_encoded'])
        
        # Save models
        joblib.dump(rf, os.path.join(model_dir, 'crime_prediction_model.pkl'))
        joblib.dump(encoder, os.path.join(model_dir, 'encoder.pkl'))
        return rf, encoder
    return None, None

def train_hotspot_model(df, model_dir):
    """Train KMeans model for crime hotspots with data validation"""
    if 'LAT' in df.columns and 'LON' in df.columns:
        # Clean data - remove invalid coordinates
        df_hotspot = df[['LAT', 'LON']].dropna()
        df_hotspot = df_hotspot[
            (df_hotspot['LAT'].between(-90, 90)) & 
            (df_hotspot['LON'].between(-180, 180))
        ]
        
        if len(df_hotspot) < 10:  # Minimum points for clustering
            return None
            
        # Train K-Means with better parameters
        kmeans = KMeans(
            n_clusters=min(10, len(df_hotspot)//10),  # Dynamic cluster count
            random_state=42,
            n_init=20,
            init='k-means++'  # Better initialization
        )
        kmeans.fit(df_hotspot)
        
        # Filter out any invalid cluster centers
        valid_centers = [
            center for center in kmeans.cluster_centers_
            if abs(center[0]) <= 90 and abs(center[1]) <= 180
        ]
        
        if not valid_centers:
            return None
            
        # Update cluster centers with only valid ones
        kmeans.cluster_centers_ = np.array(valid_centers)
        
        # Save model
        joblib.dump(kmeans, os.path.join(model_dir, 'crime_hotspot_model.pkl'))
        return kmeans
    return None

if __name__ == '__main__':
    dataset_path = "dataset/Crime_Data_from_2020_to_Present.csv"
    model_dir = "models"
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    # Load main dataset
    df = pd.read_csv(dataset_path)
    
    # Train all models
    print("Training LSTM model for crime trends...")
    crime_data = load_and_preprocess_data(dataset_path)
    lstm_model, history = train_lstm_model(crime_data, model_dir)
    
    print("\nTraining hotspot detection model...")
    hotspot_model = train_hotspot_model(df, model_dir)
    
    print("\nTraining crime type prediction model...")
    crime_pred_model, encoder = train_crime_type_model(df, model_dir)
    
    print("\n✅ All models trained and saved successfully!")