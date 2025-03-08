import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import re  # Import regular expressions

def extract_features(audio_path=None, y=None, sr=None):
    """Extract relevant audio features from the file or audio array"""
    if audio_path is not None:
        y, sr = librosa.load(audio_path)
    
    # Extract features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    
    # Example: Only return 5 features
    features = np.array([
        np.mean(mfcc),  # Mean MFCC
        np.std(mfcc),   # Standard deviation of MFCC
        np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),  # Spectral centroid mean
        np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),   # Spectral rolloff mean
        np.mean(librosa.feature.zero_crossing_rate(y=y))          # Zero crossing rate mean
    ])
    
    return features

def create_dataset():
    """Create dataset from specified voice samples"""
    data = []
    labels_authenticity = []
    labels_gender = []
    original_voice_features = []  # To store features of original voices

    # Paths for fake and real voices
    fake_female_folder = r'C:\Users\savis\Google Drive\Savitha\CAPSTONE\project - Copy\dataset\fake_voices\female'
    fake_male_folder = r'C:\Users\savis\Google Drive\Savitha\CAPSTONE\project - Copy\dataset\fake_voices\male'
    real_female_folder = r'C:\Users\savis\Google Drive\Savitha\CAPSTONE\project - Copy\dataset\real_voices\female'
    real_male_folder = r'C:\Users\savis\Google Drive\Savitha\CAPSTONE\project - Copy\dataset\real_voices\male'
    fake_folder = r'C:\Users\savis\Google Drive\Savitha\CAPSTONE\project - Copy\dataset\fake_voices\fake'
    real_folder = r'C:\Users\savis\Google Drive\Savitha\CAPSTONE\project - Copy\dataset\real_voices\real'

    # Process real male voices
    if os.path.exists(real_male_folder):
        print(f"Checking real male voices in: {real_male_folder}")
        for file in os.listdir(real_male_folder):
            if re.search(r'\.wav', file):  # Check if the file contains .wav
                features = extract_features(os.path.join(real_male_folder, file))
                data.append(features)
                labels_authenticity.append('real')
                labels_gender.append('male')
                original_voice_features.append((features, os.path.join(real_male_folder, file)))

    # Process real female voices
    if os.path.exists(real_female_folder):
        print(f"Checking real female voices in: {real_female_folder}")
        for file in os.listdir(real_female_folder):
            if re.search(r'\.wav', file):  # Check if the file contains .wav
                features = extract_features(os.path.join(real_female_folder, file))
                data.append(features)
                labels_authenticity.append('real')
                labels_gender.append('female')
                original_voice_features.append((features, os.path.join(real_female_folder, file)))

    # Process fake male voices
    if os.path.exists(fake_male_folder):
        print(f"Checking fake male voices in: {fake_male_folder}")
        for file in os.listdir(fake_male_folder):
            if re.search(r'\.wav', file):  # Check if the file contains .wav
                features = extract_features(os.path.join(fake_male_folder, file))
                data.append(features)
                labels_authenticity.append('fake')
                labels_gender.append('male')

    # Process fake female voices
    if os.path.exists(fake_female_folder):
        print(f"Checking fake female voices in: {fake_female_folder}")
        for file in os.listdir(fake_female_folder):
            if re.search(r'\.wav', file):  # Check if the file contains .wav
                features = extract_features(os.path.join(fake_female_folder, file))
                data.append(features)
                labels_authenticity.append('fake')
                labels_gender.append('female')

    # Process additional fake voices
    if os.path.exists(fake_folder):
        print(f"Checking additional fake voices in: {fake_folder}")
        for file in os.listdir(fake_folder):
            if re.search(r'\.wav', file):  # Check if the file contains .wav
                features = extract_features(os.path.join(fake_folder, file))
                data.append(features)
                labels_authenticity.append('fake')
                labels_gender.append('unknown')  # Use 'unknown' since gender is not specified

    # Process additional real voices
    if os.path.exists(real_folder):
        print(f"Checking additional real voices in: {real_folder}")
        for file in os.listdir(real_folder):
            if re.search(r'\.wav', file):  # Check if the file contains .wav
                features = extract_features(os.path.join(real_folder, file))
                data.append(features)
                labels_authenticity.append('real')
                labels_gender.append('unknown')  # Use 'unknown' since gender is not specified

    print(f"Dataset created with {len(data)} samples.")
    print(f"Real voices: {labels_authenticity.count('real')}, Fake voices: {labels_authenticity.count('fake')}")
    
    return np.array(data), np.array(labels_authenticity), np.array(labels_gender), original_voice_features

def train_models(data, labels_authenticity, labels_gender):
    """Train separate models for authenticity and gender detection"""
    # Split the data
    X_train, X_test, y_auth_train, y_auth_test, y_gender_train, y_gender_test = train_test_split(
        data, labels_authenticity, labels_gender, test_size=0.2, random_state=42
    )
    
    # Train authenticity detector
    auth_classifier = RandomForestClassifier(n_estimators=500, random_state=42)
    print("Training authenticity classifier...")
    auth_classifier.fit(X_train, y_auth_train)
    
    # Print model information
    print(f"Trained {auth_classifier.__class__.__name__} with {auth_classifier.n_estimators} estimators.")
    
    # Train gender detector
    gender_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    print("Training gender classifier...")
    gender_classifier.fit(X_train, y_gender_train)
    
    # Print model information
    print(f"Trained {gender_classifier.__class__.__name__} with {gender_classifier.n_estimators} estimators.")
    
    return auth_classifier, gender_classifier

def predict_audio(audio_path, auth_classifier, gender_classifier):
    """Predict authenticity and gender for a single audio file"""
    features = extract_features(audio_path)
    
    # Get predictions
    auth_proba = auth_classifier.predict_proba([features])[0]
    gender_proba = gender_classifier.predict_proba([features])[0]
    
    # Print probabilities for debugging
    print(f"Authenticity probabilities: {auth_proba}")
    print(f"Gender probabilities: {gender_proba}")
    
    # Adjusted thresholds for authenticity
    authenticity = 'real' if auth_proba[1] > 0.7 else 'fake'  # Adjusted threshold
    
    # Gender prediction
    if len(gender_proba) == 2:  # Check if both classes are present
        gender = 'male' if gender_proba[0] > gender_proba[1] else 'female'
    else:
        gender = 'unknown'  # If only one class is predicted
    
    return authenticity, gender

class AdvancedVoiceConverter(nn.Module):
    def __init__(self, input_dim=5):
        super(AdvancedVoiceConverter, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # First hidden layer
        self.fc2 = nn.Linear(128, 64)          # Second hidden layer
        self.fc3 = nn.Linear(64, 5)            # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def convert_voice_advanced(input_path, model, original_voice_features):
    """Convert fake voice using advanced model and find the closest original voice."""
    print("\n=== Advanced Voice Conversion Process ===")
    print(f"Converting: {os.path.basename(input_path)}")
    
    # Load and preprocess audio
    y_fake, sr = librosa.load(input_path)
    
    # Extract features
    features = extract_features(y=y_fake, sr=sr)
    features_tensor = torch.FloatTensor(features).unsqueeze(0)
    
    # Convert through model
    with torch.no_grad():
        converted = model(features_tensor)
        converted = converted.numpy().flatten()
    
    # Post-process the converted audio
    converted = post_process_audio(converted)
    
    # Match length with original
    if len(converted) < len(y_fake):
        converted = np.pad(converted, (0, len(y_fake) - len(converted)))
    else:
        converted = converted[:len(y_fake)]
    
    # Save the converted audio
    output_path = input_path.replace('.wav', '_converted.wav')
    sf.write(output_path, converted, sr)
    print(f"Successfully saved converted audio to: {output_path}")
    
    return converted, output_path

def post_process_audio(audio):
    """Apply post-processing to enhance audio quality."""
    audio = audio / np.max(np.abs(audio))  # Normalize to [-1, 1]
    return audio

def main():
    # Set the dataset path
    base_folder = r'C:\Users\savis\Google Drive\Savitha\CAPSTONE\project - Copy\dataset'
    
    # Create dataset and train classifiers
    print("Creating dataset...")
    data, labels_authenticity, labels_gender, original_voice_features = create_dataset()
    
    # Train models
    auth_classifier, gender_classifier = train_models(data, labels_authenticity, labels_gender)
    
    # Initialize voice converter
    converter = AdvancedVoiceConverter()
    
    # Process input voice file
    while True:
        input_file = input("\nEnter path to voice file (or 'q' to quit): ")
        if input_file.lower() == 'q':
            break
            
        # Remove any quotes from the path
        input_file = input_file.strip('"\'')
        
        # Check if the path exists
        if not os.path.exists(input_file):
            print("File not found. Please enter a valid path.")
            continue
            
        # Analyze voice
        print("\nAnalyzing voice...")
        authenticity, gender = predict_audio(input_file, auth_classifier, gender_classifier)
        
        print(f"Detected authenticity: {authenticity}, Detected gender: {gender}")
        
        if authenticity == 'fake':
            print("\nThis is a fake voice. Converting to original...")
            converted_audio, output_path = convert_voice_advanced(input_file, converter, original_voice_features)
            
            # Predict gender of the converted audio
            converted_authenticity, converted_gender = predict_audio(output_path, auth_classifier, gender_classifier)
            print(f"Converted audio authenticity: {converted_authenticity}, Converted audio gender: {converted_gender}")
        else:
            print("\nThis is an original voice recording. No conversion needed.")

if __name__ == "__main__":
    main()

   
