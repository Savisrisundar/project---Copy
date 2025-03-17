import React, { useState, useEffect } from 'react';
import { useLocation, Link } from 'react-router-dom';
import { Line } from 'react-chartjs-2'; // Import Chart.js for graphing
import { Chart, registerables } from 'chart.js'; // Import Chart and registerables
import './Home.css'; // Import the CSS file for styling

// Register all necessary components
Chart.register(...registerables);

const Analysis = () => {
  const location = useLocation();
  const { features, authenticity, gender, emotion } = location.state || { features: [], authenticity: '', gender: '', emotion: '' };
  const [visible, setVisible] = useState(false);
  const [activeFeature, setActiveFeature] = useState(0);

  useEffect(() => {
    // Trigger animation after component mounts
    setTimeout(() => {
      setVisible(true);
    }, 100);
  }, []);

  // Get emotion icon and color
  const getEmotionData = (emotion) => {
    switch(emotion) {
      case 'Happy/Excited': 
        return { icon: '😃', color: '#FFD700', description: 'The speaker sounds enthusiastic and positive.' };
      case 'Sad': 
        return { icon: '😢', color: '#6495ED', description: 'The speaker sounds melancholic or downcast.' };
      case 'Angry': 
        return { icon: '😠', color: '#FF6347', description: 'The speaker sounds irritated or frustrated.' };
      case 'Neutral': 
        return { icon: '😐', color: '#A9A9A9', description: 'The speaker sounds even-toned without strong emotion.' };
      case 'Calm': 
        return { icon: '😌', color: '#98FB98', description: 'The speaker sounds relaxed and composed.' };
      default: 
        return { icon: '😐', color: '#A9A9A9', description: 'Unable to determine emotional tone.' };
    }
  };

  const emotionData = getEmotionData(emotion);

  // Prepare data for the graph with units
  const featureLabels = [
    'Mean MFCC (dB)',
    'Std Dev MFCC (dB)',
    'Spectral Centroid (Hz)',
    'Spectral Rolloff (Hz)',
    'Zero Crossing Rate (crossings/sec)'
  ];
  
  // Units for each feature
  const featureUnits = ['dB', 'dB', 'Hz', 'Hz', 'crossings/sec'];
  
  // Extract numeric values from features
  const numericValues = features.map(feature => {
    if (typeof feature === 'string') {
      const match = feature.match(/:\s*(-?\d+(\.\d+)?)/);
      return match ? parseFloat(match[1]) : 0;
    }
    return feature;
  });

  const data = {
    labels: featureLabels,
    datasets: [
      {
        label: 'Audio Features',
        data: numericValues,
        borderColor: 'rgba(126, 86, 194, 0.8)',
        backgroundColor: 'rgba(74, 99, 163, 0.2)',
        fill: true,
        tension: 0.4,
        borderWidth: 3,
        pointBackgroundColor: '#7e56c2',
        pointBorderColor: '#fff',
        pointRadius: 6,
        pointHoverRadius: 8,
      },
    ],
  };

  const options = {
    scales: {
      y: {
        beginAtZero: true,
        grid: {
          color: 'rgba(255, 255, 255, 0.1)',
        },
        ticks: {
          color: 'rgba(255, 255, 255, 0.7)',
          callback: function(value) {
            return value + ' ' + featureUnits[activeFeature];
          }
        },
        title: {
          display: true,
          text: 'Value (' + featureUnits[activeFeature] + ')',
          color: 'rgba(255, 255, 255, 0.9)',
          font: {
            size: 14,
            weight: 'bold'
          }
        }
      },
      x: {
        grid: {
          color: 'rgba(255, 255, 255, 0.1)',
        },
        ticks: {
          color: 'rgba(255, 255, 255, 0.7)',
        }
      }
    },
    plugins: {
      legend: {
        labels: {
          color: 'rgba(255, 255, 255, 0.7)',
          font: {
            size: 14
          }
        }
      },
      tooltip: {
        backgroundColor: 'rgba(20, 26, 38, 0.8)',
        titleFont: {
          size: 16
        },
        bodyFont: {
          size: 14
        },
        padding: 12,
        callbacks: {
          label: function(context) {
            let label = context.dataset.label || '';
            if (label) {
              label += ': ';
            }
            if (context.parsed.y !== null) {
              const index = context.dataIndex;
              label += context.parsed.y.toFixed(2) + ' ' + featureUnits[index];
            }
            return label;
          }
        }
      }
    },
    responsive: true,
    maintainAspectRatio: false,
    onClick: (_, elements) => {
      if (elements.length > 0) {
        const index = elements[0].index;
        setActiveFeature(index);
      }
    }
  };

  return (
    <div style={{ 
      display: 'flex', 
      flexDirection: window.innerWidth < 768 ? 'column' : 'row',
      justifyContent: 'space-between', 
      marginTop: '50px', 
      padding: '0 20px',
      minHeight: '100vh',
      opacity: visible ? 1 : 0,
      transform: visible ? 'translateY(0)' : 'translateY(20px)',
      transition: 'opacity 0.8s ease, transform 0.8s ease'
    }}>
      <div style={{ width: window.innerWidth < 768 ? '100%' : '48%', marginBottom: '30px' }}>
        <h2>Analysis of Audio</h2>
        
        {/* Scale Indicator */}
        <div className="scale-indicator" style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '10px 15px',
          background: 'rgba(74, 99, 163, 0.1)',
          borderRadius: '10px',
          marginBottom: '15px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <span style={{ 
              fontSize: '20px', 
              marginRight: '10px',
              color: 'rgba(126, 86, 194, 0.8)'
            }}>📊</span>
            <span>Current Scale: <strong>{featureUnits[activeFeature]}</strong></span>
          </div>
          <div style={{ fontSize: '13px', opacity: 0.7 }}>
            Click on data points to change scale
          </div>
        </div>
        
        <div className="graph-container">
          <Line data={data} options={options} />
        </div>
        
        {/* Feature Value Cards */}
        <div style={{ 
          display: 'flex', 
          flexWrap: 'wrap', 
          gap: '10px',
          marginTop: '20px',
          marginBottom: '20px'
        }}>
          {numericValues.map((value, index) => (
            <div 
              key={index}
              style={{
                flex: '1 1 calc(33.333% - 10px)',
                minWidth: '120px',
                padding: '10px',
                background: index === activeFeature ? 'rgba(126, 86, 194, 0.2)' : 'rgba(74, 99, 163, 0.1)',
                borderRadius: '8px',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                border: index === activeFeature ? '1px solid rgba(126, 86, 194, 0.5)' : '1px solid transparent'
              }}
              onClick={() => setActiveFeature(index)}
            >
              <div style={{ fontSize: '12px', opacity: 0.7 }}>{featureLabels[index].split(' (')[0]}</div>
              <div style={{ 
                fontSize: '18px', 
                fontWeight: 'bold',
                marginTop: '5px'
              }}>
                {value.toFixed(2)} <span style={{ fontSize: '12px' }}>{featureUnits[index]}</span>
              </div>
            </div>
          ))}
        </div>
        
        <div className="features-list">
          <h3>Feature Explanation</h3>
          <ol>
            <li><strong>Mean MFCC (dB):</strong> Mel-frequency cepstral coefficients represent the short-term power spectrum of sound, indicating voice timbre.</li>
            <li><strong>Standard Deviation of MFCC (dB):</strong> Measures the variation in the MFCC, indicating voice consistency.</li>
            <li><strong>Spectral Centroid (Hz):</strong> The "center of mass" of the spectrum, indicating the brightness of the sound.</li>
            <li><strong>Spectral Rolloff (Hz):</strong> The frequency below which a specified percentage of the total spectral energy lies.</li>
            <li><strong>Zero Crossing Rate (crossings/sec):</strong> The rate at which the audio signal changes from positive to negative, indicating the noisiness of the signal.</li>
          </ol>
        </div>
      </div>
      
      <div style={{ width: window.innerWidth < 768 ? '100%' : '48%', marginBottom: '30px' }}>
        <h2>Audio Characteristics</h2>
        
        <div className="card" style={{ marginBottom: '30px', padding: '20px' }}>
          <h3>Voice Analysis</h3>
          <ul>
            <li>
              <strong>Authenticity:</strong> 
              <span style={{ 
                display: 'inline-block',
                marginLeft: '10px',
                padding: '5px 12px',
                borderRadius: '20px',
                background: authenticity === 'fake' ? 'rgba(255, 152, 0, 0.2)' : 'rgba(76, 175, 80, 0.2)',
                color: authenticity === 'fake' ? '#ff9800' : '#4CAF50',
                fontWeight: 'bold'
              }}>
                {authenticity}
              </span>
            </li>
            <li>
              <strong>Gender:</strong> 
              <span style={{ 
                display: 'inline-block',
                marginLeft: '10px',
                padding: '5px 12px',
                borderRadius: '20px',
                background: 'rgba(74, 99, 163, 0.2)',
                color: '#7e56c2',
                fontWeight: 'bold'
              }}>
                {gender}
              </span>
            </li>
            <li><strong>Conversion Method:</strong> The model converts fake voices to original voices by analyzing audio features and applying transformations based on trained data.</li>
            <li><strong>Features Used:</strong> {features.join(', ')}</li>
          </ul>
        </div>
        
        <div className="card" style={{ marginBottom: '30px', padding: '20px' }}>
          <h3>Emotional Analysis</h3>
          <div style={{ 
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            marginBottom: '20px'
          }}>
            <div style={{
              fontSize: '48px',
              margin: '10px 0',
            }}>
              {emotionData.icon}
            </div>
            <div style={{
              fontSize: '24px',
              fontWeight: 'bold',
              color: emotionData.color,
              margin: '10px 0',
            }}>
              {emotion}
            </div>
          </div>
          
          <p style={{ textAlign: 'left', lineHeight: '1.6' }}>
            {emotionData.description}
          </p>
          
          <div style={{ 
            marginTop: '20px',
            padding: '15px',
            borderRadius: '10px',
            background: 'rgba(74, 99, 163, 0.1)',
          }}>
            <h4 style={{ marginTop: 0 }}>Emotion Indicators</h4>
            <ul style={{ textAlign: 'left' }}>
              <li><strong>Pitch Variation:</strong> {emotion === 'Happy/Excited' || emotion === 'Angry' ? 'High' : 'Low'}</li>
              <li><strong>Speech Rate:</strong> {emotion === 'Happy/Excited' || emotion === 'Angry' ? 'Fast' : emotion === 'Sad' ? 'Slow' : 'Medium'}</li>
              <li><strong>Voice Intensity:</strong> {emotion === 'Happy/Excited' || emotion === 'Angry' ? 'Strong' : emotion === 'Sad' ? 'Weak' : 'Moderate'}</li>
            </ul>
          </div>
        </div>
        
        <div style={{ marginTop: '40px', textAlign: 'center' }}>
          <Link to="/" style={{ textDecoration: 'none' }}>
            <button className="button">Back to Home</button>
          </Link>
        </div>
      </div>
    </div>
  );
};

export default Analysis;
