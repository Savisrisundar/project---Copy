import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import './Home.css'; // Import the CSS file for styling

const Home = () => {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    // Trigger animation after component mounts
    setVisible(true);
  }, []);

  return (
    <div style={{ 
      textAlign: 'center', 
      marginTop: '50px',
      padding: '0 20px',
      opacity: visible ? 1 : 0,
      transition: 'opacity 1s ease-in-out'
    }}>
      <div className="card" style={{ 
        maxWidth: '700px', 
        margin: '0 auto',
        marginBottom: '40px'
      }}>
        <h1>Voice Conversion App</h1>
        
        <p className="subtitle">
          Detect and convert fake voices to their original form with advanced AI technology
        </p>
        
        <div className="features-list">
          <h3>Features</h3>
          <ul>
            <li>Detect if a voice recording is authentic or fake</li>
            <li>Convert fake voices back to their original form</li>
            <li>Analyze voice characteristics and emotional tone</li>
            <li>Record your voice directly in the app</li>
            <li>Support for multiple audio formats</li>
          </ul>
        </div>
        
        <div style={{ marginTop: '30px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '15px' }}>
          <Link to="/upload" style={{ textDecoration: 'none', width: '80%', maxWidth: '300px' }}>
            <button className="button">Upload Audio File</button>
          </Link>
          <Link to="/recorder" style={{ textDecoration: 'none', width: '80%', maxWidth: '300px' }}>
            <button className="button">Record Your Voice</button>
          </Link>
        </div>
      </div>
      
      <div style={{ 
        marginTop: '60px', 
        fontSize: '14px', 
        opacity: '0.7',
        maxWidth: '700px',
        margin: '60px auto 0'
      }}>
        <p>Powered by advanced machine learning algorithms for voice authenticity detection and conversion.</p>
      </div>
    </div>
  );
};

export default Home;
