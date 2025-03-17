import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';

import Home from './components/Home';
import Upload from './components/Upload';
import Analysis from './components/Analysis';
import Results from './components/Results';
import VoiceRecorder from './components/VoiceRecorder';
import Cursor from './components/Cursor';
import './components/Home.css';

function App() {
  return (
    <Router>
      <div>
        {/* Animated Background */}
        <div className="animated-background"></div>
        
        {/* Custom Cursor */}
        <Cursor />
        
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/upload" element={<Upload />} />
          <Route path="/analysis" element={<Analysis />} />
          <Route path="/results" element={<Results />} />
          <Route path="/recorder" element={<VoiceRecorder />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
