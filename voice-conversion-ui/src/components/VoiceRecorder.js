import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import './Home.css';

// Remove the unused global variables
// let timerInterval = null;
// let timerStartTime = 0;

const VoiceRecorder = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [audioBlob, setAudioBlob] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [recordingComplete, setRecordingComplete] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [visualizerData, setVisualizerData] = useState(Array(50).fill(2));
  const [audioUrl, setAudioUrl] = useState(null);
  
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const animationFrameRef = useRef(null);
  const audioRef = useRef(null);
  const streamRef = useRef(null);
  const navigate = useNavigate();

  // Add a ref for the timer element
  const timerDisplayRef = useRef(null);
  
  // Timer variables
  const timerIntervalRef = useRef(null);
  const timerStartTimeRef = useRef(0);

  // Safely close audio context
  const safelyCloseAudioContext = useCallback(() => {
    if (audioContextRef.current && 
        audioContextRef.current.state !== 'closed' && 
        audioContextRef.current.state !== 'closing') {
      try {
        audioContextRef.current.close().catch(err => {
          console.log('Error closing AudioContext:', err);
        });
      } catch (err) {
        console.log('Exception when closing AudioContext:', err);
      }
    }
  }, []);

  // Use useCallback to memoize the stopRecording function
  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      try {
        mediaRecorderRef.current.stop();
        console.log("Recording stopped successfully");
      } catch (err) {
        console.log('Error stopping media recorder:', err);
      }
      
      // Clear the timer interval
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current);
        timerIntervalRef.current = null;
      }
      
      setIsRecording(false);
      
      // Stop all tracks on the stream
      if (streamRef.current) {
        try {
          streamRef.current.getTracks().forEach(track => track.stop());
        } catch (err) {
          console.log('Error stopping tracks:', err);
        }
      }
      
      // Stop visualizer
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
    }
  }, [isRecording]);

  // Visualizer update function - completely revised for mobile-like waves
  const updateVisualizer = useCallback(() => {
    if (!analyserRef.current) return;
    
    const bufferLength = analyserRef.current.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    const updateData = () => {
      if (!analyserRef.current) return;
      
      try {
        analyserRef.current.getByteFrequencyData(dataArray);
        
        // For a more mobile-like wave visualization
        const numBars = 60; // More bars for smoother wave
        const sampledData = [];
        
        // Calculate average volume for scaling
        let totalVolume = 0;
        for (let i = 0; i < bufferLength; i++) {
          totalVolume += dataArray[i];
        }
        const averageVolume = totalVolume / bufferLength;
        
        // Create a wave-like pattern
        for (let i = 0; i < numBars; i++) {
          // Sample from the middle frequencies for better visualization
          const startIndex = Math.floor(bufferLength * 0.2);
          const endIndex = Math.floor(bufferLength * 0.8);
          const range = endIndex - startIndex;
          
          // Get a value from the frequency data
          const index = startIndex + Math.floor((i / numBars) * range);
          let value = dataArray[index] || 0;
          
          // Apply dynamic scaling based on recording state
          if (isRecording) {
            // More dramatic scaling when recording
            const scaleFactor = 0.7 + (averageVolume / 255) * 1.5;
            value = value * scaleFactor;
            
            // Add some randomness for natural movement
            value += (Math.random() * 5) - 2.5;
          } else {
            // Gentle ambient movement when not recording
            value = Math.sin(Date.now() * 0.001 + i * 0.2) * 10 + 15;
          }
          
          // Ensure minimum height
          value = Math.max(3, value);
          
          sampledData.push(value);
        }
        
        // Smooth the wave by averaging neighboring bars
        const smoothedData = [];
        for (let i = 0; i < sampledData.length; i++) {
          const prev = i > 0 ? sampledData[i-1] : sampledData[i];
          const current = sampledData[i];
          const next = i < sampledData.length - 1 ? sampledData[i+1] : sampledData[i];
          
          // Weighted average for smoother transitions
          const smoothed = (prev * 0.3) + (current * 0.4) + (next * 0.3);
          smoothedData.push(smoothed);
        }
        
        setVisualizerData(smoothedData);
      } catch (err) {
        console.log('Error updating visualizer:', err);
      }
      
      animationFrameRef.current = requestAnimationFrame(updateData);
    };
    
    updateData();
  }, [isRecording]);

  useEffect(() => {
    // Clean up on component unmount
    return () => {
      if (isRecording) {
        stopRecording();
      }
      
      // Clear the timer interval
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current);
        timerIntervalRef.current = null;
      }
      
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
      
      // Safely close AudioContext
      safelyCloseAudioContext();
      
      // Clean up audio URL
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
    };
  }, [stopRecording, safelyCloseAudioContext, isRecording, audioUrl]);

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const startRecording = async () => {
    try {
      // Reset state
      setErrorMessage('');
      audioChunksRef.current = [];
      
      // Get user media
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      
      // Create audio context for visualization
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      audioContextRef.current = audioContext;
      
      // Create analyzer for visualization
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      analyserRef.current = analyser;
      
      // Connect the microphone to the analyzer
      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);
      
      // Start the visualizer
      updateVisualizer();
      
      // Create media recorder with proper MIME type
      try {
        // Try to use WebM format which is widely supported
        const options = { mimeType: 'audio/webm' };
        mediaRecorderRef.current = new MediaRecorder(stream, options);
        console.log("Using audio/webm format");
      } catch (e) {
        console.log('WebM not supported, trying MP3');
        try {
          const options = { mimeType: 'audio/mp3' };
          mediaRecorderRef.current = new MediaRecorder(stream, options);
          console.log("Using audio/mp3 format");
        } catch (e2) {
          console.log('No specific format supported, using default');
          mediaRecorderRef.current = new MediaRecorder(stream);
        }
      }
      
      // Set up recorder events
      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      mediaRecorderRef.current.onstop = () => {
        // Create blob from recorded chunks
        const audioBlob = new Blob(audioChunksRef.current, { 
          type: audioChunksRef.current[0].type 
        });
        
        // Create URL for the audio blob
        const audioUrl = URL.createObjectURL(audioBlob);
        
        // Update state with the recorded audio
        setAudioBlob(audioBlob);
        setAudioUrl(audioUrl);
        setRecordingComplete(true);
        
        console.log(`Recording completed: ${audioBlob.size} bytes, type: ${audioBlob.type}`);
      };
      
      // Start recording
      mediaRecorderRef.current.start();
      setIsRecording(true);
      
      // Start timer
      timerStartTimeRef.current = Date.now();
      timerIntervalRef.current = setInterval(() => {
        const elapsedSeconds = Math.floor((Date.now() - timerStartTimeRef.current) / 1000);
        setRecordingTime(elapsedSeconds);
      }, 1000);
      
    } catch (err) {
      console.error('Error starting recording:', err);
      setErrorMessage('Could not access microphone. Please check permissions.');
    }
  };

  const handleSubmit = async () => {
    if (!audioBlob || audioBlob.size === 0) {
      setErrorMessage('Please record audio first');
      return;
    }
    
    setIsProcessing(true);
    setErrorMessage('');
    
    try {
      console.log(`Preparing to upload audio blob of size: ${audioBlob.size}`);
      
      // Get the actual MIME type from the blob
      const mimeType = audioBlob.type || 'audio/webm';
      console.log(`Audio MIME type: ${mimeType}`);
      
      // Use the correct file extension based on the MIME type
      const fileExt = mimeType.includes('webm') ? '.webm' : 
                     mimeType.includes('mp3') ? '.mp3' : '.wav';
      const fileName = `recording_${Date.now()}${fileExt}`;
      
      // Create a FormData object
      const formData = new FormData();
      
      // Use the original blob with its correct extension
      const audioFile = new File([audioBlob], fileName, { type: mimeType });
      
      formData.append('file', audioFile);
      
      console.log(`Uploading file: ${fileName} (${audioFile.size} bytes) with type ${mimeType}`);
      
      // Send to server
      const response = await axios.post('http://localhost:5000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      console.log("Server response:", response.data);
      
      if (response.data.download_url) {
        const downloadUrl = `http://localhost:5000${response.data.download_url}`;
        const originalUrl = response.data.original_url ? 
          `http://localhost:5000${response.data.original_url}` : '';
        
        // Navigate to results page
        navigate('/results', { 
          state: { 
            outputMessage: response.data.output,
            downloadUrl: downloadUrl,
            originalUrl: originalUrl,
            fullPath: response.data.full_path,
            features: response.data.features,
            authenticity: response.data.authenticity,
            gender: response.data.gender,
            emotion: response.data.emotion
          } 
        });
      } else {
        setErrorMessage('Server did not return processed audio. Please try again.');
      }
    } catch (error) {
      console.error('Error uploading audio:', error);
      if (error.response) {
        console.error('Server error details:', error.response.data);
        setErrorMessage(`Server error: ${error.response.data.error || 'Unknown error'}`);
      } else {
        setErrorMessage('Error processing audio. Please try again.');
      }
    } finally {
      setIsProcessing(false);
    }
  };

  // Add this function to convert AudioBuffer to WAV
  const audioBufferToWav = (buffer) => {
    return new Promise((resolve) => {
      const numberOfChannels = buffer.numberOfChannels;
      const sampleRate = buffer.sampleRate;
      const length = buffer.length * numberOfChannels * 2;
      
      // Create WAV header
      const wav = new ArrayBuffer(44 + length);
      const view = new DataView(wav);
      
      // "RIFF" chunk descriptor
      writeString(view, 0, 'RIFF');
      view.setUint32(4, 36 + length, true);
      writeString(view, 8, 'WAVE');
      
      // "fmt " sub-chunk
      writeString(view, 12, 'fmt ');
      view.setUint32(16, 16, true); // subchunk size
      view.setUint16(20, 1, true); // PCM format
      view.setUint16(22, numberOfChannels, true);
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * numberOfChannels * 2, true); // byte rate
      view.setUint16(32, numberOfChannels * 2, true); // block align
      view.setUint16(34, 16, true); // bits per sample
      
      // "data" sub-chunk
      writeString(view, 36, 'data');
      view.setUint32(40, length, true);
      
      // Write the PCM samples
      const offset = 44;
      let pos = offset;
      
      for (let i = 0; i < buffer.length; i++) {
        for (let channel = 0; channel < numberOfChannels; channel++) {
          const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]));
          const int16 = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
          view.setInt16(pos, int16, true);
          pos += 2;
        }
      }
      
      // Create Blob and resolve
      const wavBlob = new Blob([wav], { type: 'audio/wav' });
      resolve(wavBlob);
    });
  };

  // Helper function to write strings to DataView
  const writeString = (view, offset, string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };

  const playRecording = () => {
    if (audioRef.current && audioUrl) {
      audioRef.current.play();
    }
  };

  return (
    <div className="container">
      <h2>Live Voice Recorder</h2>
      
      <div className="recorder-container">
        {/* Audio Visualizer - updated for mobile-like wave */}
        <div className="visualizer-container" style={{
          height: '150px',
          background: 'rgba(10, 14, 23, 0.7)',
          borderRadius: '10px',
          padding: '10px',
          marginBottom: '20px',
          display: 'flex',
          alignItems: 'center', // Center vertically for wave effect
          justifyContent: 'center',
          position: 'relative',
          overflow: 'hidden'
        }}>
          <div className="wave-container" style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: '100%',
            height: '100%',
            position: 'relative'
          }}>
            {visualizerData.map((value, index) => (
              <div 
                key={index} 
                className="wave-bar"
                style={{ 
                  height: `${value}%`,
                  backgroundColor: isRecording ? 
                    `hsl(${210 + (value/2)}, 80%, 60%)` : 
                    `hsl(240, 60%, 60%)`,
                  width: '3px',
                  borderRadius: '3px',
                  margin: '0 1px',
                  transition: 'height 0.05s ease-in-out',
                  transform: 'scaleY(1)',
                  opacity: isRecording ? 1 : 0.6,
                  boxShadow: isRecording ? '0 0 8px rgba(100, 150, 255, 0.5)' : 'none'
                }}
              />
            ))}
          </div>
        </div>
        
        {/* Recording Timer - updated with more prominent styling */}
        <div className="recording-timer">
          {isRecording ? (
            <div className="recording-indicator">
              <span className="recording-dot"></span>
              <span>Recording</span>
            </div>
          ) : recordingComplete ? (
            <div className="recording-complete">Recording Complete</div>
          ) : (
            <div>Ready to Record</div>
          )}
          <div 
            ref={timerDisplayRef} 
            className="timer"
            style={{
              fontSize: '28px',
              fontWeight: 'bold',
              color: isRecording ? '#ff4b4b' : 'inherit'
            }}
          >
            {formatTime(recordingTime)}
          </div>
        </div>
        
        {/* Controls */}
        <div className="recorder-controls">
          {!isRecording ? (
            <button 
              className="button" 
              onClick={startRecording}
              disabled={isProcessing}
            >
              Start Recording
            </button>
          ) : (
            <button 
              className="button button-stop" 
              onClick={stopRecording}
            >
              Stop Recording
            </button>
          )}
          
          {recordingComplete && audioUrl && (
            <>
              <button 
                className="button button-play" 
                onClick={playRecording}
              >
                Play Recording
              </button>
              <button 
                className="button" 
                onClick={handleSubmit}
                disabled={isProcessing}
              >
                {isProcessing ? 'Processing...' : 'Process Recording'}
              </button>
            </>
          )}
        </div>
        
        {/* Hidden audio element for playback */}
        {audioUrl && (
          <audio 
            ref={audioRef} 
            src={audioUrl} 
            style={{ display: 'none' }} 
            controls={false}
          />
        )}
        
        {errorMessage && (
          <div className="error-message">{errorMessage}</div>
        )}
        
        <div className="recorder-instructions">
          <p>Click "Start Recording" to begin capturing your voice. When finished, click "Stop Recording" and then "Process Recording" to analyze.</p>
        </div>
      </div>
      
      {/* Add wave animation styles */}
      <style>
        {`
          @keyframes wave {
            0%, 100% { transform: scaleY(1); }
            50% { transform: scaleY(1.1); }
          }
          
          .wave-bar {
            animation: wave 1.5s infinite ease-in-out;
            animation-delay: calc(var(--i) * 0.05s);
          }
          
          .recording-dot {
            display: inline-block;
            width: 12px;
            height: 12px;
            background-color: #ff4b4b;
            border-radius: 50%;
            margin-right: 8px;
            animation: blink 1s infinite;
          }
          
          @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
          }
          
          .timer {
            transition: color 0.3s ease;
          }
        `}
      </style>
    </div>
  );
};

export default VoiceRecorder; 