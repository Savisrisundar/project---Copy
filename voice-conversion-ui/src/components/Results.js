import React, { useState, useEffect, useRef } from 'react';
import { useLocation, Link } from 'react-router-dom';
import './Home.css'; // Import the CSS file for styling

const Results = () => {
  const location = useLocation();
  const { outputMessage, downloadUrl, originalUrl, fullPath, features, authenticity, gender, emotion } = location.state || { 
    outputMessage: '', 
    downloadUrl: '', 
    originalUrl: '',
    fullPath: '', 
    features: [], 
    authenticity: '', 
    gender: '',
    emotion: ''
  };
  const [showPath, setShowPath] = useState(false); // State to control visibility of the file path
  const [visible, setVisible] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isOriginalPlaying, setIsOriginalPlaying] = useState(false);
  const audioRef = useRef(null);
  const originalAudioRef = useRef(null);

  useEffect(() => {
    // Trigger animation after component mounts
    setTimeout(() => {
      setVisible(true);
    }, 100);
  }, []);

  const handleDownloadClick = () => {
    setShowPath(true); // Show the file path when the download link is clicked
  };

  const handlePlayClick = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleOriginalPlayClick = () => {
    if (originalAudioRef.current) {
      if (isOriginalPlaying) {
        originalAudioRef.current.pause();
      } else {
        originalAudioRef.current.play();
      }
      setIsOriginalPlaying(!isOriginalPlaying);
    }
  };

  // Handle audio ended event
  const handleAudioEnded = () => {
    setIsPlaying(false);
  };

  const handleOriginalAudioEnded = () => {
    setIsOriginalPlaying(false);
  };

  // Get emotion icon
  const getEmotionIcon = (emotion) => {
    switch(emotion) {
      case 'Happy/Excited': return '😃';
      case 'Sad': return '😢';
      case 'Angry': return '😠';
      case 'Neutral': return '😐';
      case 'Calm': return '😌';
      default: return '😐';
    }
  };

  // Format features for better understanding
  const formattedFeatures = features.map((feature, index) => {
    switch (index) {
      case 0: return `Mean MFCC: ${feature}`;
      case 1: return `Standard Deviation of MFCC: ${feature}`;
      case 2: return `Spectral Centroid: ${feature}`;
      case 3: return `Spectral Rolloff: ${feature}`;
      case 4: return `Zero Crossing Rate: ${feature}`;
      default: return `Feature ${index + 1}: ${feature}`;
    }
  });

  // Log the features before navigating
  console.log("Features being passed to Analysis:", formattedFeatures);

  return (
    <div style={{ 
      textAlign: 'center', 
      marginTop: '50px', 
      padding: '0 20px',
      opacity: visible ? 1 : 0,
      transform: visible ? 'translateY(0)' : 'translateY(20px)',
      transition: 'opacity 0.8s ease, transform 0.8s ease'
    }}>
      <h1>Your Results</h1>
      
      <div className="card" style={{ 
        maxWidth: '600px',
        margin: '0 auto 30px auto',
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          width: '70px',
          height: '70px',
          borderRadius: '50%',
          background: authenticity === 'fake' ? 'rgba(255, 152, 0, 0.2)' : 'rgba(76, 175, 80, 0.2)',
          margin: '0 auto 20px',
        }}>
          <span style={{
            fontSize: '30px',
            color: authenticity === 'fake' ? '#ff9800' : '#4CAF50',
          }}>
            {authenticity === 'fake' ? '!' : '✓'}
          </span>
        </div>
        
        <p style={{ fontSize: '18px', marginBottom: '25px' }}>{outputMessage}</p>
        
        {/* Emotion Display */}
        {emotion && (
          <div className="emotion-container" style={{
            marginBottom: '20px',
            padding: '10px',
            background: 'rgba(74, 99, 163, 0.1)',
            borderRadius: '10px',
          }}>
            <h3>Emotional Tone</h3>
            <div style={{
              fontSize: '32px',
              margin: '10px 0',
            }}>
              {getEmotionIcon(emotion)}
            </div>
            <p>{emotion}</p>
          </div>
        )}
        
        {/* Audio Players */}
        <div style={{ marginBottom: '20px' }}>
          {originalUrl && (
            <div className="audio-player-container">
              <h3>Original Audio</h3>
              <audio 
                ref={originalAudioRef} 
                src={originalUrl} 
                onEnded={handleOriginalAudioEnded} 
                style={{ display: 'none' }}
              />
              <button 
                className="button audio-button" 
                onClick={handleOriginalPlayClick}
              >
                {isOriginalPlaying ? 'Pause' : 'Play Original'}
              </button>
            </div>
          )}
          
          {downloadUrl && (
            <div className="audio-player-container">
              <h3>{authenticity === 'fake' ? 'Converted Audio' : 'Audio'}</h3>
              <audio 
                ref={audioRef} 
                src={downloadUrl} 
                onEnded={handleAudioEnded} 
                style={{ display: 'none' }}
              />
              <button 
                className="button audio-button" 
                onClick={handlePlayClick}
              >
                {isPlaying ? 'Pause' : 'Play'}
              </button>
            </div>
          )}
        </div>
        
        {downloadUrl && (
          <a href={downloadUrl} className="download-link" onClick={handleDownloadClick} download>
            Download {authenticity === 'fake' ? 'Converted' : ''} Audio
          </a>
        )}
        
        {showPath && fullPath && (
          <p style={{ marginTop: '20px', fontSize: '14px', opacity: '0.7' }}>File saved at: {fullPath}</p>
        )}
      </div>
      
      <div style={{ marginTop: '30px', display: 'flex', justifyContent: 'center', gap: '20px' }}>
        <Link to="/analysis" state={{ features: formattedFeatures, authenticity, gender, emotion }} style={{ textDecoration: 'none' }}>
          <button className="button">View Analysis</button>
        </Link>
        
        <Link to="/" style={{ textDecoration: 'none' }}>
          <button className="button button-transparent">Back to Home</button>
        </Link>
      </div>
    </div>
  );
};

export default Results; 