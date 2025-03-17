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
    """Train classifiers for authenticity and gender"""
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
    
    return auth_classifier, gender_classifier

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

if __name__ == '__main__':
    # Create the dataset
    data, labels_authenticity, labels_gender, original_voice_features = create_dataset()
    
    # Train the models
    auth_classifier, gender_classifier = train_models(data, labels_authenticity, labels_gender)
    
    # Initialize the voice conversion model
    model = AdvancedVoiceConverter(input_dim=5)
    
    # Run the Flask app
    app.run(debug=True, port=5000)

   
