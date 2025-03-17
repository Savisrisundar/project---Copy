import React, { useState, useEffect } from 'react';
import './Home.css';

const Cursor = () => {
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [hidden, setHidden] = useState(false);
  const [clicking, setClicking] = useState(false);
  const [linkHover, setLinkHover] = useState(false);

  useEffect(() => {
    // Add event listener for mouse movement
    const addEventListeners = () => {
      document.addEventListener('mousemove', onMouseMove);
      document.addEventListener('mouseenter', onMouseEnter);
      document.addEventListener('mouseleave', onMouseLeave);
      document.addEventListener('mousedown', onMouseDown);
      document.addEventListener('mouseup', onMouseUp);
    };

    // Remove event listeners
    const removeEventListeners = () => {
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseenter', onMouseEnter);
      document.removeEventListener('mouseleave', onMouseLeave);
      document.removeEventListener('mousedown', onMouseDown);
      document.removeEventListener('mouseup', onMouseUp);
    };

    // Track cursor position
    const onMouseMove = (e) => {
      setPosition({ x: e.clientX, y: e.clientY });
    };

    // Show cursor when mouse enters window
    const onMouseEnter = () => {
      setHidden(false);
    };

    // Hide cursor when mouse leaves window
    const onMouseLeave = () => {
      setHidden(true);
    };

    // Click animation
    const onMouseDown = () => {
      setClicking(true);
    };

    const onMouseUp = () => {
      setClicking(false);
    };

    // Check for hover over interactive elements
    const handleLinkHoverEvents = () => {
      document.querySelectorAll('a, button, input, .download-link, .features-list li, ul li').forEach(el => {
        el.addEventListener('mouseenter', () => setLinkHover(true));
        el.addEventListener('mouseleave', () => setLinkHover(false));
      });
    };

    addEventListeners();
    handleLinkHoverEvents();

    return () => {
      removeEventListeners();
    };
  }, []);

  useEffect(() => {
    // Create cursor trail effect
    const createTrail = () => {
      const trail = document.createElement('div');
      trail.className = 'cursor-trail';
      trail.style.left = `${position.x}px`;
      trail.style.top = `${position.y}px`;
      document.body.appendChild(trail);
      
      // Animate and remove the trail dot
      trail.style.animation = 'fadeOut 0.8s ease forwards';
      setTimeout(() => {
        document.body.removeChild(trail);
      }, 800);
    };
    
    // Create trail dots at intervals
    let trailInterval;
    if (!hidden && !linkHover) {
      trailInterval = setInterval(createTrail, 100);
    }
    
    return () => {
      clearInterval(trailInterval);
    };
  }, [position, hidden, linkHover]);

  return (
    <>
      <div
        className={`cursor-dot ${clicking ? 'clicking' : ''} ${linkHover ? 'link-hover' : ''}`}
        style={{
          left: `${position.x}px`,
          top: `${position.y}px`,
          opacity: hidden ? 0 : 1,
        }}
      />
      <div
        className={`cursor-outline ${clicking ? 'clicking' : ''} ${linkHover ? 'link-hover' : ''}`}
        style={{
          left: `${position.x}px`,
          top: `${position.y}px`,
          opacity: hidden ? 0 : 1,
        }}
      />
    </>
  );
};

export default Cursor; 