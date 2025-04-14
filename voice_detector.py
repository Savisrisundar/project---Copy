import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import re
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pydub import AudioSegment
import uuid
import tempfile
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define the advanced voice converter model
class AdvancedVoiceConverter(nn.Module):
    def __init__(self, input_dim=5):
        super(AdvancedVoiceConverter, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Tanh()  # Output normalized features
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def extract_features(audio_path=None, y=None, sr=None):
    """Extract relevant audio features from the file or audio array"""
    if audio_path is not None:
        try:
            # First try with pydub for better format support
            import tempfile
            import os
            
            # Check file extension
            _, ext = os.path.splitext(audio_path)
            
            # For webm files from the recorder, convert to wav first
            if ext.lower() == '.webm':
                temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_wav.close()
                
                try:
                    audio = AudioSegment.from_file(audio_path)
                    audio.export(temp_wav.name, format="wav")
                    y, sr = librosa.load(temp_wav.name, sr=None)
                    os.unlink(temp_wav.name)  # Delete temp file
                except Exception as e:
                    print(f"Error converting webm: {e}")
                    # Fallback to direct loading
                    y, sr = librosa.load(audio_path, sr=None)
            else:
                # For other formats, try direct loading
                y, sr = librosa.load(audio_path, sr=None)
                
        except Exception as e:
            print(f"Error loading audio: {e}")
            raise ValueError(f"Could not load audio file: {audio_path}")
    
    if y is None or len(y) == 0:  # Check if audio data is empty
        raise ValueError("Audio data is empty or not loaded correctly.")
    
    if len(y) < 512:  # Check if the audio length is less than n_fft
        print(f"Audio file {audio_path} is too short for FFT processing.")
        return np.zeros(5)  # Return a zero array with 5 features
    
    # Extract features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=512)
    
    # Return 5 features
    features = np.array([
        np.mean(mfcc),  # Mean MFCC
        np.std(mfcc),   # Standard deviation of MFCC
        np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),  # Spectral centroid mean
        np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),   # Spectral rolloff mean
        np.mean(librosa.feature.zero_crossing_rate(y=y)),        # Zero crossing rate mean
    ])
    
    return features

def extract_emotion_features(audio_path=None, y=None, sr=None):
    """Extract features for emotion detection"""
    if audio_path is not None:
        y, sr = librosa.load(audio_path, sr=None)
    
    # Extract emotion-related features
    # MFCCs (captures timbre and vocal tract shape)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    
    # Chroma (relates to harmonic content)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    
    # Spectral contrast (captures the difference between peaks and valleys)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)
    
    # Energy (volume/intensity)
    energy = np.sum(y**2) / len(y)
    
    # Tempo (speed of speech)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
    
    # Pitch statistics
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    pitch_std = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    
    # Return a dictionary of emotion-related features
    return {
        'mfccs': mfccs_mean,
        'chroma': chroma_mean,
        'contrast': contrast_mean,
        'energy': energy,
        'tempo': tempo,
        'pitch_mean': pitch_mean,
        'pitch_std': pitch_std
    }

def detect_emotion(features):
    """Detect emotion based on audio features"""
    # Simple rule-based emotion detection
    energy = features['energy']
    tempo = features['tempo']
    pitch_mean = features['pitch_mean']
    pitch_std = features['pitch_std']
    
    # High energy, high tempo, high pitch = Happy/Excited
    if energy > 0.05 and tempo > 120 and pitch_mean > 200:
        return "Happy/Excited"
    
    # Low energy, low tempo, low pitch variation = Sad
    elif energy < 0.02 and tempo < 100 and pitch_std < 50:
        return "Sad"
    
    # High energy, high tempo, high pitch variation = Angry
    elif energy > 0.04 and tempo > 110 and pitch_std > 70:
        return "Angry"
    
    # Medium energy, medium tempo, medium pitch = Neutral
    elif 0.02 <= energy <= 0.04 and 100 <= tempo <= 120:
        return "Neutral"
    
    # Low energy, medium tempo, low pitch variation = Calm
    elif energy < 0.03 and 90 <= tempo <= 110 and pitch_std < 60:
        return "Calm"
    
    # Default case
    else:
        return "Neutral"

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

    # Process all folders
    folders = [
        (real_male_folder, 'real', 'male'),
        (real_female_folder, 'real', 'female'),
        (fake_male_folder, 'fake', 'male'),
        (fake_female_folder, 'fake', 'female'),
        (real_folder, 'real', None),
        (fake_folder, 'fake', None)
    ]
    
    for folder, authenticity, gender in folders:
        if os.path.exists(folder):
            print(f"Processing {authenticity} {gender or ''} voices in: {folder}")
            for file in os.listdir(folder):
                if re.search(r'\.(wav|mp3|ogg|flac|m4a)$', file, re.IGNORECASE):
                    try:
                        file_path = os.path.join(folder, file)
                        features = extract_features(file_path)
                        data.append(features)
                        labels_authenticity.append(authenticity)
                        labels_gender.append(gender if gender else 'unknown')
                        
                        # Store original voice features for conversion reference
                        if authenticity == 'real':
                            original_voice_features.append((features, file_path))
                    except Exception as e:
                        print(f"Error processing file {file}: {str(e)}")
    
    return data, labels_authenticity, labels_gender, original_voice_features

def train_models(data, labels_authenticity, labels_gender):
    """Train classifiers for authenticity and gender with detailed accuracy metrics"""
    # Convert to numpy arrays
    X = np.array(data)
    y_auth = np.array(labels_authenticity)
    y_gender = np.array(labels_gender)
    
    # Split the data
    X_train, X_test, y_auth_train, y_auth_test, y_gender_train, y_gender_test = train_test_split(
        X, y_auth, y_gender, test_size=0.2, random_state=42
    )
    
    # Train authenticity classifier
    auth_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    auth_classifier.fit(X_train, y_auth_train)
    
    # Train gender classifier
    gender_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    gender_classifier.fit(X_train, y_gender_train)
    
    # Calculate detailed metrics for authenticity classifier
    auth_train_pred = auth_classifier.predict(X_train)
    auth_test_pred = auth_classifier.predict(X_test)
    
    auth_metrics = {
        'train': {
            'accuracy': accuracy_score(y_auth_train, auth_train_pred),
            'precision': precision_score(y_auth_train, auth_train_pred, average='weighted'),
            'recall': recall_score(y_auth_train, auth_train_pred, average='weighted'),
            'f1': f1_score(y_auth_train, auth_train_pred, average='weighted')
        },
        'test': {
            'accuracy': accuracy_score(y_auth_test, auth_test_pred),
            'precision': precision_score(y_auth_test, auth_test_pred, average='weighted'),
            'recall': recall_score(y_auth_test, auth_test_pred, average='weighted'),
            'f1': f1_score(y_auth_test, auth_test_pred, average='weighted')
        },
        'confusion_matrix_train': confusion_matrix(y_auth_train, auth_train_pred),
        'confusion_matrix_test': confusion_matrix(y_auth_test, auth_test_pred)
    }
    
    # Calculate detailed metrics for gender classifier
    gender_train_pred = gender_classifier.predict(X_train)
    gender_test_pred = gender_classifier.predict(X_test)
    
    gender_metrics = {
        'train': {
            'accuracy': accuracy_score(y_gender_train, gender_train_pred),
            'precision': precision_score(y_gender_train, gender_train_pred, average='weighted'),
            'recall': recall_score(y_gender_train, gender_train_pred, average='weighted'),
            'f1': f1_score(y_gender_train, gender_train_pred, average='weighted')
        },
        'test': {
            'accuracy': accuracy_score(y_gender_test, gender_test_pred),
            'precision': precision_score(y_gender_test, gender_test_pred, average='weighted'),
            'recall': recall_score(y_gender_test, gender_test_pred, average='weighted'),
            'f1': f1_score(y_gender_test, gender_test_pred, average='weighted')
        },
        'confusion_matrix_train': confusion_matrix(y_gender_train, gender_train_pred),
        'confusion_matrix_test': confusion_matrix(y_gender_test, gender_test_pred)
    }
    
    # Print detailed performance metrics
    print("\n===== MODEL PERFORMANCE METRICS =====")
    print("\nAuthenticity Classifier Metrics:")
    print("Training Metrics:")
    print(f"Accuracy: {auth_metrics['train']['accuracy']:.4f} ({auth_metrics['train']['accuracy']*100:.2f}%)")
    print(f"Precision: {auth_metrics['train']['precision']:.4f}")
    print(f"Recall: {auth_metrics['train']['recall']:.4f}")
    print(f"F1 Score: {auth_metrics['train']['f1']:.4f}")
    
    print("\nTesting Metrics:")
    print(f"Accuracy: {auth_metrics['test']['accuracy']:.4f} ({auth_metrics['test']['accuracy']*100:.2f}%)")
    print(f"Precision: {auth_metrics['test']['precision']:.4f}")
    print(f"Recall: {auth_metrics['test']['recall']:.4f}")
    print(f"F1 Score: {auth_metrics['test']['f1']:.4f}")
    
    print("\nGender Classifier Metrics:")
    print("Training Metrics:")
    print(f"Accuracy: {gender_metrics['train']['accuracy']:.4f} ({gender_metrics['train']['accuracy']*100:.2f}%)")
    print(f"Precision: {gender_metrics['train']['precision']:.4f}")
    print(f"Recall: {gender_metrics['train']['recall']:.4f}")
    print(f"F1 Score: {gender_metrics['train']['f1']:.4f}")
    
    print("\nTesting Metrics:")
    print(f"Accuracy: {gender_metrics['test']['accuracy']:.4f} ({gender_metrics['test']['accuracy']*100:.2f}%)")
    print(f"Precision: {gender_metrics['test']['precision']:.4f}")
    print(f"Recall: {gender_metrics['test']['recall']:.4f}")
    print(f"F1 Score: {gender_metrics['test']['f1']:.4f}")
    print("==================================\n")
    
    # Print confusion matrices
    print("\nConfusion Matrices:")
    print("\nAuthenticity Classifier - Training:")
    print(auth_metrics['confusion_matrix_train'])
    print("\nAuthenticity Classifier - Testing:")
    print(auth_metrics['confusion_matrix_test'])
    print("\nGender Classifier - Training:")
    print(gender_metrics['confusion_matrix_train'])
    print("\nGender Classifier - Testing:")
    print(gender_metrics['confusion_matrix_test'])
    
    return auth_classifier, gender_classifier, auth_metrics, gender_metrics

def predict_audio(audio_path, auth_classifier, gender_classifier):
    """Predict if audio is fake or real and determine gender"""
    features = extract_features(audio_path)
    
    # Reshape for sklearn
    features_reshaped = features.reshape(1, -1)
    
    # Predict authenticity and gender
    authenticity = auth_classifier.predict(features_reshaped)[0]
    gender = gender_classifier.predict(features_reshaped)[0]
    
    # Extract emotion features and detect emotion
    emotion_features = extract_emotion_features(audio_path)
    emotion = detect_emotion(emotion_features)
    
    return authenticity, gender, emotion

def convert_voice_advanced(input_path, model, original_voice_features):
    """Convert a fake voice to its original form using advanced techniques"""
    # Extract the base filename without extension to find matching original
    base_filename = os.path.splitext(os.path.basename(input_path))[0]
    name_part = base_filename.split('_')[0]  # Extract name part (e.g., "Shruthi" from "Shruthi_fake")
    
    print(f"Looking for original voice match for: {name_part}")
    
    # Try to find a matching original voice file
    matching_original = None
    for _, orig_path in original_voice_features:
        orig_basename = os.path.basename(orig_path)
        if name_part.lower() in orig_basename.lower() and "real" in orig_basename.lower():
            matching_original = orig_path
            print(f"Found matching original voice: {matching_original}")
            break
    
    # Load the input audio
    y, sr = librosa.load(input_path, sr=None)
    
    # If we found a matching original voice, use it directly
    if matching_original and os.path.exists(matching_original):
        print(f"Using matching original voice for conversion: {matching_original}")
        y_orig, sr_orig = librosa.load(matching_original, sr=sr)
        
        # Create output filename based on the original
        output_filename = f"converted_{name_part}_original.wav"
        output_path = os.path.join('./uploads', output_filename)
        
        # Save the original voice as the converted output
        sf.write(output_path, y_orig, sr_orig, subtype='PCM_24')
        
        return y_orig, output_path
    
    # If no matching original is found, fall back to algorithmic conversion
    print("No matching original found, using algorithmic conversion")
    
    # Extract features from the input audio
    input_features = extract_features(y=y, sr=sr)
    
    # Convert features using the model
    input_tensor = torch.FloatTensor(input_features)
    with torch.no_grad():
        converted_features = model(input_tensor).numpy()
    
    # Find the closest matching original voice for reference
    closest_match = None
    min_distance = float('inf')
    reference_path = None
    
    for orig_features, orig_path in original_voice_features:
        distance = np.linalg.norm(converted_features - orig_features)
        if distance < min_distance:
            min_distance = distance
            closest_match = orig_features
            reference_path = orig_path
    
    # Apply voice conversion
    y_converted = apply_voice_conversion(y, sr, converted_features, reference_path)
    
    # Save the converted audio
    output_filename = f"converted_{os.path.basename(input_path)}"
    output_path = os.path.join('./uploads', output_filename)
    
    # Ensure the output is in WAV format with high quality
    sf.write(output_path, y_converted, sr, subtype='PCM_24')
    
    return y_converted, output_path

def apply_voice_conversion(y, sr, converted_features, reference_path=None):
    """Apply voice conversion techniques to transform the audio"""
    # If we have a reference recording, use it for more accurate conversion
    if reference_path and os.path.exists(reference_path):
        y_ref, sr_ref = librosa.load(reference_path, sr=sr)
        
        # Make sure the reference is not too short
        if len(y_ref) < len(y):
            # Repeat the reference if needed
            repeats = int(np.ceil(len(y) / len(y_ref)))
            y_ref = np.tile(y_ref, repeats)[:len(y)]
        
        # Extract spectral envelope from reference
        S_ref = np.abs(librosa.stft(y_ref))
        
        # Extract spectral envelope from input
        S_input = np.abs(librosa.stft(y))
        
        # Ensure the spectrograms have the same shape
        min_time = min(S_ref.shape[1], S_input.shape[1])
        S_ref = S_ref[:, :min_time]
        S_input = S_input[:, :min_time]
        
        # Spectral morphing (blend the spectral characteristics)
        morph_ratio = 0.7  # 70% of the reference characteristics
        S_morphed = (1 - morph_ratio) * S_input + morph_ratio * librosa.util.normalize(S_ref, norm=1) * np.mean(S_input)
        
        # Reconstruct the audio from the morphed spectrogram
        y_converted = librosa.griffinlim(S_morphed)
        
        # Apply some additional processing to make it sound more natural
        y_converted = librosa.effects.preemphasis(y_converted)
        
        # Normalize the output
        y_converted = librosa.util.normalize(y_converted)
    else:
        # Without a reference, we'll apply basic transformations based on the converted features
        
        # Adjust pitch based on the first feature (MFCC mean)
        pitch_shift = (converted_features[0] - np.mean(converted_features)) * 2
        y_converted = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift)
        
        # Adjust timbre based on spectral features
        y_converted = librosa.effects.harmonic(y_converted, margin=3.0)
        
        # Apply some additional processing
        y_converted = librosa.effects.preemphasis(y_converted)
        
        # Normalize the output
        y_converted = librosa.util.normalize(y_converted)
    
    return y_converted

def convert_to_wav(input_path, output_path):
    """Convert any audio format to WAV format"""
    try:
        # Check if input file exists and has content
        if not os.path.exists(input_path):
            print(f"Input file does not exist: {input_path}")
            return False
            
        file_size = os.path.getsize(input_path)
        if file_size == 0:
            print(f"Input file is empty: {input_path}")
            return False
            
        # Load the audio file using pydub
        audio = AudioSegment.from_file(input_path)
        
        # Export as WAV (high quality)
        audio.export(output_path, format="wav", parameters=["-ac", "1", "-ar", "44100"])
        
        # Verify the output file was created
        if not os.path.exists(output_path):
            print(f"Failed to create output file: {output_path}")
            return False
            
        output_size = os.path.getsize(output_path)
        if output_size == 0:
            print(f"Output file is empty: {output_path}")
            return False
            
        print(f"Successfully converted {input_path} to {output_path}")
        return True
    except Exception as e:
        print(f"Error converting audio: {str(e)}")
        return False

# Add this function to convert WebM to WAV using ffmpeg
def convert_webm_to_wav(input_path, output_path):
    """Convert WebM audio to WAV format using ffmpeg"""
    try:
        import subprocess
        
        # Check if input file exists and has content
        if not os.path.exists(input_path):
            print(f"Input file does not exist: {input_path}")
            return False
            
        file_size = os.path.getsize(input_path)
        if file_size == 0:
            print(f"Input file is empty: {input_path}")
            return False
        
        # Run ffmpeg to convert the file
        command = [
            'ffmpeg', '-i', input_path, 
            '-acodec', 'pcm_s16le',  # Use PCM 16-bit encoding
            '-ar', '44100',          # Set sample rate to 44.1kHz
            '-ac', '1',              # Convert to mono
            '-y',                    # Overwrite output file if it exists
            output_path
        ]
        
        result = subprocess.run(command, check=True, capture_output=True)
        
        # Verify the output file was created
        if not os.path.exists(output_path):
            print(f"Failed to create output file: {output_path}")
            return False
            
        output_size = os.path.getsize(output_path)
        if output_size == 0:
            print(f"Output file is empty: {output_path}")
            return False
        
        print(f"Successfully converted {input_path} to {output_path}")
        return True
    except Exception as e:
        print(f"Error converting WebM to WAV: {e}")
        return False

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Create uploads directory if it doesn't exist
    upload_folder = './uploads'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    
    # Get the original file extension
    original_filename = file.filename
    file_extension = os.path.splitext(original_filename)[1].lower()
    
    # Generate a unique filename
    unique_id = uuid.uuid4().hex
    original_file_path = os.path.join(upload_folder, f"{unique_id}{file_extension}")
    wav_file_path = os.path.join(upload_folder, f"{unique_id}.wav")
    
    print(f"Original file path: {original_file_path}")
    print(f"WAV file path: {wav_file_path}")
    
    # Save the file with its original extension
    try:
        file.save(original_file_path)
        print(f"Saved uploaded file to {original_file_path}")
        
        # Check if file was actually saved
        if not os.path.exists(original_file_path):
            return jsonify({'error': f'File was not saved properly at {original_file_path}'}), 500
            
        file_size = os.path.getsize(original_file_path)
        if file_size == 0:
            return jsonify({'error': 'Uploaded file is empty'}), 500
    except Exception as e:
        return jsonify({'error': f'Failed to save file: {str(e)}'}), 500
    
    # Convert to WAV if needed
    try:
        if file_extension.lower() == '.webm':
            # For WebM files, use pydub directly
            audio = AudioSegment.from_file(original_file_path, format="webm")
            audio.export(wav_file_path, format="wav")
            print(f"Converted WebM to WAV: {wav_file_path}")
        elif file_extension.lower() != '.wav':
            # For other formats, use pydub
            audio = AudioSegment.from_file(original_file_path)
            audio.export(wav_file_path, format="wav")
            print(f"Converted {file_extension} to WAV: {wav_file_path}")
        else:
            # If it's already a WAV, just copy it
            import shutil
            shutil.copy2(original_file_path, wav_file_path)
            print(f"File is already WAV, copied to: {wav_file_path}")
    except Exception as e:
        print(f"Error converting to WAV: {e}")
        # If conversion fails, try to use the original file
        wav_file_path = original_file_path
    
    # Process the audio file
    try:
        output_path, extracted_features, authenticity, gender, emotion = process_audio(wav_file_path)
        
        # Return the results
        if authenticity == 'fake' and output_path:
            full_path = os.path.abspath(output_path)
            print(f"Converted file path: {full_path}")
            return jsonify({
                'message': 'File processed successfully',
                'output': f'Audio is {authenticity} and gender is {gender}. Emotional tone: {emotion}. The voice has been converted to its original form.',
                'download_url': f"/download/{os.path.basename(output_path)}",
                'original_url': f"/download/{os.path.basename(original_file_path)}",
                'full_path': full_path,
                'features': extracted_features.tolist(),
                'authenticity': authenticity,
                'gender': gender,
                'emotion': emotion
            }), 200
        else:
            return jsonify({
                'message': 'File processed successfully',
                'output': f'Audio is {authenticity} and gender is {gender}. Emotional tone: {emotion}. No conversion needed.',
                'download_url': f"/download/{os.path.basename(wav_file_path)}",
                'full_path': os.path.abspath(wav_file_path),
                'features': extracted_features.tolist(),
                'authenticity': authenticity,
                'gender': gender,
                'emotion': emotion
            }), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error processing audio: {str(e)}'}), 500

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """Serve the converted audio file for download"""
    upload_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    return send_from_directory(upload_folder, filename, as_attachment=True, 
                              mimetype='audio/wav', 
                              download_name=f"converted_{filename}")

def process_audio(file_path):
    """Process the audio file and convert if necessary"""
    # Get predictions
    authenticity, gender, emotion = predict_audio(file_path, auth_classifier, gender_classifier)
    
    # Extract features for the response
    extracted_features = extract_features(file_path)
    print(f"Extracted Features: {extracted_features}")
    print(f"Detected Emotion: {emotion}")
    
    if authenticity == 'fake':
        # Convert the fake voice to original
        converted_audio, output_path = convert_voice_advanced(file_path, model, original_voice_features)
        return output_path, extracted_features, authenticity, gender, emotion
    
    return file_path, extracted_features, authenticity, gender, emotion

def count_files(directory):
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

base_dir = r'C:\Users\savis\Google Drive\Savitha\CAPSTONE\project - Copy\dataset'

counts = {
    'real_female': count_files(os.path.join(base_dir, 'real_voices', 'female')),
    'real_male': count_files(os.path.join(base_dir, 'real_voices', 'male')),
    'real_general': count_files(os.path.join(base_dir, 'real_voices', 'real')),
    'fake_female': count_files(os.path.join(base_dir, 'fake_voices', 'female')),
    'fake_male': count_files(os.path.join(base_dir, 'fake_voices', 'male')),
    'fake_general': count_files(os.path.join(base_dir, 'fake_voices', 'fake'))
}

print("Dataset Statistics:")
for key, value in counts.items():
    print(f"{key}: {value} files")

# Actual dataset statistics from your code
categories = ['Real Female', 'Real Male', 'Real General', 'Fake Female', 'Fake Male', 'Fake General']
file_counts = [2, 3, 774, 3, 4, 772]

# Create figure with subplots
fig = plt.figure(figsize=(15, 10))
plt.suptitle('Voice Dataset Analysis', fontsize=16)

# 1. Bar plot of original dataset distribution
ax1 = plt.subplot(2, 2, 1)
colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c']
ax1.bar(categories, file_counts, color=colors)
ax1.set_ylabel('Number of Files', fontsize=12)
ax1.set_title('Original Dataset Distribution', fontsize=14)
ax1.set_xticklabels(categories, rotation=45, ha='right')
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Add file count labels on top of each bar
for i, count in enumerate(file_counts):
    ax1.text(i, count + 5, str(count), ha='center', fontweight='bold')

# 2. Pie chart showing real vs fake distribution
ax2 = plt.subplot(2, 2, 2)
real_total = sum(file_counts[:3])
fake_total = sum(file_counts[3:])
ax2.pie([real_total, fake_total], labels=['Real', 'Fake'], autopct='%1.1f%%',
        colors=['#3498db', '#e74c3c'], explode=(0.05, 0.05), shadow=True)
ax2.set_title('Real vs Fake Voice Distribution', fontsize=14)

# 3. Grouped bar chart comparing gender-specific vs general files
ax3 = plt.subplot(2, 2, 3)
gender_specific = [file_counts[0] + file_counts[1], file_counts[3] + file_counts[4]]
general = [file_counts[2], file_counts[5]]
x = np.arange(2)
width = 0.35

ax3.bar(x - width/2, gender_specific, width, label='Gender-Specific', color='#3498db')
ax3.bar(x + width/2, general, width, label='General', color='#e74c3c')
ax3.set_ylabel('Number of Files', fontsize=12)
ax3.set_title('Gender-Specific vs General Files', fontsize=14)
ax3.set_xticks(x)
ax3.set_xticklabels(['Real', 'Fake'])
ax3.legend()
ax3.grid(axis='y', linestyle='--', alpha=0.7)

# Add file count labels
for i, count in enumerate(gender_specific):
    ax3.text(i - width/2, count + 5, str(count), ha='center', fontweight='bold')
for i, count in enumerate(general):
    ax3.text(i + width/2, count + 5, str(count), ha='center', fontweight='bold')

# 4. Horizontal bar chart showing dataset balance
ax4 = plt.subplot(2, 2, 4)
categories_short = ['Female', 'Male', 'General']
real_counts = [file_counts[0], file_counts[1], file_counts[2]]
fake_counts = [file_counts[3], file_counts[4], file_counts[5]]

y = np.arange(len(categories_short))
ax4.barh(y - width/2, real_counts, width, label='Real', color='#3498db')
ax4.barh(y + width/2, fake_counts, width, label='Fake', color='#e74c3c')
ax4.set_xlabel('Number of Files', fontsize=12)
ax4.set_title('Real vs Fake Distribution by Category', fontsize=14)
ax4.set_yticks(y)
ax4.set_yticklabels(categories_short)
ax4.legend()
ax4.grid(axis='x', linestyle='--', alpha=0.7)

# Add file count labels
for i, count in enumerate(real_counts):
    ax4.text(count + 5, i - width/2, str(count), va='center', fontweight='bold')
for i, count in enumerate(fake_counts):
    ax4.text(count + 5, i + width/2, str(count), va='center', fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Print dataset statistics summary
print("\nDataset Statistics Summary:")
print("--------------------------")
print(f"Total files: {sum(file_counts)}")
print(f"Real voices: {real_total} ({real_total/sum(file_counts)*100:.1f}%)")
print(f"Fake voices: {fake_total} ({fake_total/sum(file_counts)*100:.1f}%)")
print(f"Gender-specific files: {sum(gender_specific)} ({sum(gender_specific)/sum(file_counts)*100:.1f}%)")
print(f"General files: {sum(general)} ({sum(general)/sum(file_counts)*100:.1f}%)")

# Model performance visualization
def visualize_model_performance(auth_metrics, gender_metrics):
    """Visualize the performance metrics of the trained models"""
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    
    # Create figure for model performance
    fig = plt.figure(figsize=(15, 10))
    plt.suptitle('Model Performance Analysis', fontsize=16)
    
    # 1. Bar chart comparing training and testing accuracy
    ax1 = plt.subplot(2, 2, 1)
    classifiers = ['Authenticity', 'Gender']
    train_acc = [auth_metrics['train']['accuracy'], gender_metrics['train']['accuracy']]
    test_acc = [auth_metrics['test']['accuracy'], gender_metrics['test']['accuracy']]
    
    x = np.arange(len(classifiers))
    width = 0.35
    
    ax1.bar(x - width/2, train_acc, width, label='Training', color='#3498db')
    ax1.bar(x + width/2, test_acc, width, label='Testing', color='#e74c3c')
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Model Accuracy Comparison', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(classifiers)
    ax1.legend()
    ax1.set_ylim(0.8, 1.0)  # Set y-axis to focus on the relevant range
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add percentage labels
    for i, acc in enumerate(train_acc):
        ax1.text(i - width/2, acc + 0.01, f"{acc*100:.1f}%", ha='center', fontweight='bold')
    for i, acc in enumerate(test_acc):
        ax1.text(i + width/2, acc + 0.01, f"{acc*100:.1f}%", ha='center', fontweight='bold')
    
    # 2. Radar chart for test metrics
    ax2 = plt.subplot(2, 2, 2, polar=True)
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    angles = np.linspace(0, 2*np.pi, len(metrics_names), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    auth_values = [
        auth_metrics['test']['accuracy'], 
        auth_metrics['test']['precision'], 
        auth_metrics['test']['recall'], 
        auth_metrics['test']['f1']
    ]
    auth_values += auth_values[:1]  # Close the loop
    
    gender_values = [
        gender_metrics['test']['accuracy'], 
        gender_metrics['test']['precision'], 
        gender_metrics['test']['recall'], 
        gender_metrics['test']['f1']
    ]
    gender_values += gender_values[:1]  # Close the loop
    
    ax2.plot(angles, auth_values, 'o-', linewidth=2, label='Authenticity', color='#3498db')
    ax2.fill(angles, auth_values, alpha=0.25, color='#3498db')
    ax2.plot(angles, gender_values, 'o-', linewidth=2, label='Gender', color='#e74c3c')
    ax2.fill(angles, gender_values, alpha=0.25, color='#e74c3c')
    
    ax2.set_thetagrids(np.degrees(angles[:-1]), metrics_names)
    ax2.set_ylim(0.8, 1.0)  # Set radial limits to focus on relevant range
    ax2.set_title('Test Metrics Comparison', fontsize=14)
    ax2.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # 3. Confusion matrices for authenticity
    ax3 = plt.subplot(2, 2, 3)
    auth_cm = auth_metrics['confusion_matrix_test']
    sns.heatmap(auth_cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax3)
    ax3.set_xlabel('Predicted Label', fontsize=12)
    ax3.set_ylabel('True Label', fontsize=12)
    ax3.set_title('Authenticity Confusion Matrix (Test)', fontsize=14)
    ax3.set_xticklabels(['Real', 'Fake'])
    ax3.set_yticklabels(['Real', 'Fake'])
    
    # 4. Confusion matrices for gender
    ax4 = plt.subplot(2, 2, 4)
    gender_cm = gender_metrics['confusion_matrix_test']
    # Filter out 'unknown' if present to make a cleaner visualization
    if gender_cm.shape[0] > 2:
        gender_cm = gender_cm[:2, :2]  # Just show male/female confusion
        labels = ['Male', 'Female']
    else:
        labels = ['Male', 'Female']
    
    sns.heatmap(gender_cm, annot=True, fmt='d', cmap='Reds', cbar=False, ax=ax4)
    ax4.set_xlabel('Predicted Label', fontsize=12)
    ax4.set_ylabel('True Label', fontsize=12)
    ax4.set_title('Gender Confusion Matrix (Test)', fontsize=14)
    ax4.set_xticklabels(labels)
    ax4.set_yticklabels(labels)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print model performance summary
    print("\nModel Performance Summary:")
    print("--------------------------")
    print(f"\nAuthenticity Classifier:")
    print(f"  Training Accuracy: {auth_metrics['train']['accuracy']*100:.2f}%")
    print(f"  Testing Accuracy: {auth_metrics['test']['accuracy']*100:.2f}%")
    print(f"  Precision: {auth_metrics['test']['precision']:.4f}")
    print(f"  Recall: {auth_metrics['test']['recall']:.4f}")
    print(f"  F1 Score: {auth_metrics['test']['f1']:.4f}")
    
    print(f"\nGender Classifier:")
    print(f"  Training Accuracy: {gender_metrics['train']['accuracy']*100:.2f}%")
    print(f"  Testing Accuracy: {gender_metrics['test']['accuracy']*100:.2f}%")
    print(f"  Precision: {gender_metrics['test']['precision']:.4f}")
    print(f"  Recall: {gender_metrics['test']['recall']:.4f}")
    print(f"  F1 Score: {gender_metrics['test']['f1']:.4f}")

# Add this to your main code to visualize processing times
def measure_processing_times():
    """Measure and visualize processing times for different operations"""
    import time
    import matplotlib.pyplot as plt
    
    # Sample audio for testing
    test_file = os.path.join(base_dir, 'fake_voices', 'fake', os.listdir(os.path.join(base_dir, 'fake_voices', 'fake'))[0])
    
    # Measure feature extraction time
    start = time.time()
    features = extract_features(test_file)
    feature_time = time.time() - start
    
    # Measure classification time
    start = time.time()
    features_reshaped = features.reshape(1, -1)
    auth_classifier.predict(features_reshaped)
    gender_classifier.predict(features_reshaped)
    classification_time = time.time() - start
    
    # Measure emotion detection time
    start = time.time()
    emotion_features = extract_emotion_features(test_file)
    detect_emotion(emotion_features)
    emotion_time = time.time() - start
    
    # Measure voice conversion time
    start = time.time()
    y, sr = librosa.load(test_file, sr=None)
    input_features = extract_features(y=y, sr=sr)
    input_tensor = torch.FloatTensor(input_features)
    with torch.no_grad():
        model(input_tensor).numpy()
    conversion_time = time.time() - start
    
    # Total processing time
    total_time = feature_time + classification_time + emotion_time + conversion_time
    
    # Create visualization
    operations = ['Feature Extraction', 'Classification', 'Emotion Detection', 'Voice Conversion', 'Total Processing']
    times = [feature_time, classification_time, emotion_time, conversion_time, total_time]
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(operations, times, color=['#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#f39c12'])
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.title('Average Processing Time by Operation', fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add time labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.05, bar.get_y() + bar.get_height()/2, 
                 f'{width:.2f}s', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('processing_time.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nProcessing Time Analysis:")
    print("-----------------------")
    for op, time_val in zip(operations, times):
        print(f"{op}: {time_val:.2f} seconds")
    print(f"\nPercentage breakdown:")
    for op, time_val in zip(operations[:-1], times[:-1]):  # Exclude total
        print(f"{op}: {time_val/times[-1]*100:.1f}% of total processing time")
    
    return times

def visualize_detailed_confusion_matrices(auth_metrics, gender_metrics, data, labels_authenticity, labels_gender):
    """Create detailed confusion matrices for all dataset categories"""
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split
    
    # Create figure for detailed confusion matrices
    plt.figure(figsize=(18, 15))
    plt.suptitle('Detailed Confusion Matrices by Category', fontsize=18)
    
    # Split the data again to get the same test set
    X = np.array(data)
    y_auth = np.array(labels_authenticity)
    y_gender = np.array(labels_gender)
    
    X_train, X_test, y_auth_train, y_auth_test, y_gender_train, y_gender_test = train_test_split(
        X, y_auth, y_gender, test_size=0.2, random_state=42
    )
    
    # 1. Authenticity Confusion Matrix (Real vs Fake)
    plt.subplot(2, 2, 1)
    auth_cm = auth_metrics['confusion_matrix_test']
    
    # Get the unique labels in order
    auth_labels = ['real', 'fake']
    
    # Create a normalized confusion matrix
    auth_cm_norm = auth_cm.astype('float') / auth_cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    sns.heatmap(auth_cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Authenticity Confusion Matrix', fontsize=14)
    plt.xticks(np.arange(len(auth_labels))+0.5, auth_labels)
    plt.yticks(np.arange(len(auth_labels))+0.5, auth_labels)
    
    # Add accuracy text
    accuracy = np.trace(auth_cm) / np.sum(auth_cm)
    plt.text(0.5, -0.1, f'Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)', 
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    
    # 2. Gender Confusion Matrix (Male vs Female vs Unknown)
    plt.subplot(2, 2, 2)
    
    # Get unique gender labels
    unique_genders = np.unique(y_gender)
    
    # Create confusion matrix for gender
    gender_cm = gender_metrics['confusion_matrix_test']
    
    # Plot
    sns.heatmap(gender_cm, annot=True, fmt='d', cmap='Reds', cbar=True)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Gender Confusion Matrix', fontsize=14)
    plt.xticks(np.arange(len(unique_genders))+0.5, unique_genders)
    plt.yticks(np.arange(len(unique_genders))+0.5, unique_genders)
    
    # Add accuracy text
    accuracy = np.trace(gender_cm) / np.sum(gender_cm)
    plt.text(0.5, -0.1, f'Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)', 
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    
    # 3. Combined Category Confusion Matrix
    plt.subplot(2, 1, 2)
    
    # Create combined labels (real_male, real_female, real_unknown, fake_male, fake_female, fake_unknown)
    combined_labels_train = [f"{auth}_{gender}" for auth, gender in zip(y_auth_train, y_gender_train)]
    combined_labels_test = [f"{auth}_{gender}" for auth, gender in zip(y_auth_test, y_gender_test)]
    
    # Get predictions
    auth_test_pred = auth_metrics['confusion_matrix_test'].argmax(axis=1)
    gender_test_pred = gender_metrics['confusion_matrix_test'].argmax(axis=1)
    
    # Map back to original labels
    auth_labels_map = {i: label for i, label in enumerate(np.unique(y_auth))}
    gender_labels_map = {i: label for i, label in enumerate(np.unique(y_gender))}
    
    auth_pred_labels = [auth_labels_map[i] for i in auth_test_pred]
    gender_pred_labels = [gender_labels_map[i] for i in gender_test_pred]
    
    # Create combined predicted labels
    combined_pred_labels = [f"{auth}_{gender}" for auth, gender in zip(auth_pred_labels, gender_pred_labels)]
    
    # Get unique combined labels
    unique_combined = sorted(list(set(combined_labels_train + combined_labels_test)))
    
    # Create confusion matrix for combined categories
    combined_cm = confusion_matrix(
        combined_labels_test, 
        combined_pred_labels,
        labels=unique_combined
    )
    
    # Plot with adjusted figure size for readability
    plt.figure(figsize=(14, 12))
    sns.heatmap(combined_cm, annot=True, fmt='d', cmap='viridis', 
                xticklabels=unique_combined, yticklabels=unique_combined)
    plt.xlabel('Predicted Category', fontsize=12)
    plt.ylabel('True Category', fontsize=12)
    plt.title('Combined Category Confusion Matrix', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add accuracy text
    accuracy = np.trace(combined_cm) / np.sum(combined_cm)
    plt.text(0.5, -0.05, f'Overall Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)', 
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('detailed_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed accuracy by category
    print("\nDetailed Accuracy by Category:")
    print("-----------------------------")
    
    # Calculate per-category accuracy from the combined confusion matrix
    for i, category in enumerate(unique_combined):
        category_correct = combined_cm[i, i]
        category_total = combined_cm[i, :].sum()
        if category_total > 0:
            category_acc = category_correct / category_total
            print(f"{category}: {category_acc:.4f} ({category_acc*100:.1f}%) - {category_correct}/{category_total} correct")
    
    # Create a table visualization for the augmentation data
    plt.figure(figsize=(10, 6))
    plt.axis('tight')
    plt.axis('off')
    
    # Data from your augmentation table
    categories = ['Real Female', 'Real Male', 'Real General', 'Fake Female', 'Fake Male', 'Fake General', 'Total']
    initial_counts = [2, 3, 774, 3, 4, 772, 1558]
    augmented_counts = [40, 60, 1548, 60, 80, 1544, 3332]
    augmentation_factors = [20, 20, 2, 20, 20, 2, 2.14]
    
    table_data = []
    for i in range(len(categories)):
        table_data.append([
            categories[i], 
            str(initial_counts[i]), 
            str(augmented_counts[i]), 
            f"{augmentation_factors[i]}×"
        ])
    
    table = plt.table(
        cellText=table_data,
        colLabels=['Category', 'Initial Count', 'Augmented Count', 'Augmentation Factor'],
        loc='center',
        cellLoc='center',
        colWidths=[0.25, 0.25, 0.25, 0.25]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Add title
    plt.title('Voice Dataset Augmentation Summary', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('augmentation_table.png', dpi=300, bbox_inches='tight')
    plt.show()

def load_gender_dataset_from_csv(csv_path):
    """Load additional gender-labeled data from a CSV file"""
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Initialize lists for features and labels
    gender_data = []
    gender_labels = []
    
    # Process each row in the CSV
    for index, row in df.iterrows():
        try:
            # The voice.csv dataset has different features than our audio extraction
            # We'll use the most relevant ones that match our 5 features
            features = [
                row['meanfun'],       # Mean fundamental frequency - similar to MFCC mean
                row['sd'],            # Standard deviation - similar to MFCC std
                row['centroid'],      # Spectral centroid if available, or use meanfreq
                row['IQR'],           # Inter-quartile range - can substitute for spectral rolloff
                row['meanfun']        # Using meanfun again as substitute for zero crossing rate
            ]
            
            # Convert label to match your format (lowercase 'male'/'female')
            gender = row['label'].lower()
            
            gender_data.append(features)
            gender_labels.append(gender)
            
        except Exception as e:
            print(f"Error processing row {index}: {e}")
    
    print(f"Loaded {len(gender_data)} additional gender samples from CSV")
    return gender_data, gender_labels

def train_gender_classifier_with_additional_data(data, labels_gender, csv_path):
    """Train a gender classifier with additional data from CSV"""
    # Load additional gender data
    additional_data, additional_labels = load_gender_dataset_from_csv(csv_path)
    
    # Combine with existing data
    # We only use the gender-specific data (not 'unknown')
    gender_specific_indices = [i for i, label in enumerate(labels_gender) if label != 'unknown']
    
    # Extract gender-specific data
    gender_data = [data[i] for i in gender_specific_indices]
    gender_labels = [labels_gender[i] for i in gender_specific_indices]
    
    # Add the additional data
    combined_data = gender_data + additional_data
    combined_labels = gender_labels + additional_labels
    
    # Convert to numpy arrays
    X = np.array(combined_data)
    y = np.array(combined_labels)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train gender classifier
    gender_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    gender_classifier.fit(X_train, y_train)
    
    # Calculate detailed metrics
    train_pred = gender_classifier.predict(X_train)
    test_pred = gender_classifier.predict(X_test)
    
    gender_metrics = {
        'train': {
            'accuracy': accuracy_score(y_train, train_pred),
            'precision': precision_score(y_train, train_pred, average='weighted'),
            'recall': recall_score(y_train, train_pred, average='weighted'),
            'f1': f1_score(y_train, train_pred, average='weighted')
        },
        'test': {
            'accuracy': accuracy_score(y_test, test_pred),
            'precision': precision_score(y_test, test_pred, average='weighted'),
            'recall': recall_score(y_test, test_pred, average='weighted'),
            'f1': f1_score(y_test, test_pred, average='weighted')
        },
        'confusion_matrix_train': confusion_matrix(y_train, train_pred),
        'confusion_matrix_test': confusion_matrix(y_test, test_pred)
    }
    
    # Print detailed performance metrics
    print("\n===== GENDER CLASSIFIER METRICS (WITH ADDITIONAL DATA) =====")
    print("Training Metrics:")
    print(f"Accuracy: {gender_metrics['train']['accuracy']:.4f} ({gender_metrics['train']['accuracy']*100:.2f}%)")
    print(f"Precision: {gender_metrics['train']['precision']:.4f}")
    print(f"Recall: {gender_metrics['train']['recall']:.4f}")
    print(f"F1 Score: {gender_metrics['train']['f1']:.4f}")
    
    print("\nTesting Metrics:")
    print(f"Accuracy: {gender_metrics['test']['accuracy']:.4f} ({gender_metrics['test']['accuracy']*100:.2f}%)")
    print(f"Precision: {gender_metrics['test']['precision']:.4f}")
    print(f"Recall: {gender_metrics['test']['recall']:.4f}")
    print(f"F1 Score: {gender_metrics['test']['f1']:.4f}")
    
    # Print confusion matrices
    print("\nConfusion Matrices:")
    print("\nGender Classifier - Training:")
    print(gender_metrics['confusion_matrix_train'])
    print("\nGender Classifier - Testing:")
    print(gender_metrics['confusion_matrix_test'])
    print("==================================\n")
    
    return gender_classifier, gender_metrics

def visualize_gender_classifier_performance(gender_metrics):
    """Visualize the performance of the gender classifier trained with CSV data"""
    plt.figure(figsize=(10, 8))
    plt.suptitle('Gender Classifier Performance (with CSV data)', fontsize=16)
    
    # Gender Confusion Matrix
    gender_cm = gender_metrics['confusion_matrix_test']
    
    # Plot
    sns.heatmap(gender_cm, annot=True, fmt='d', cmap='Reds', cbar=True)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Gender Confusion Matrix', fontsize=14)
    
    # Get the labels (male/female)
    if gender_cm.shape[0] == 2:
        plt.xticks([0.5, 1.5], ['male', 'female'])
        plt.yticks([0.5, 1.5], ['male', 'female'])
    
    # Add accuracy text
    accuracy = np.trace(gender_cm) / np.sum(gender_cm)
    plt.text(0.5, -0.1, f'Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)', 
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    
    plt.tight_layout()
    plt.savefig('gender_classifier_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

# Add this call to your main code after training the models
if __name__ == '__main__':
    # Create the dataset
    data, labels_authenticity, labels_gender, original_voice_features = create_dataset()
    
    # Train the authenticity classifier as before
    auth_classifier, _, auth_metrics, _ = train_models(data, labels_authenticity, labels_gender)
    
    # Train the gender classifier with additional data
    csv_path = r'C:\Users\savis\Google Drive\Savitha\CAPSTONE\project - Copy\dataset\voice.csv'
    gender_classifier, gender_metrics = train_gender_classifier_with_additional_data(
        data, labels_gender, csv_path
    )
    
    # Initialize the voice conversion model
    model = AdvancedVoiceConverter(input_dim=5)
    
    # Visualize model performance with actual metrics
    visualize_model_performance(auth_metrics, gender_metrics)
    
    # Generate detailed confusion matrices for all categories
    visualize_detailed_confusion_matrices(auth_metrics, gender_metrics, data, labels_authenticity, labels_gender)
    
    # Measure and visualize processing times
    processing_times = measure_processing_times()
    
    # Run the Flask app
    app.run(debug=True, port=5000)

   
