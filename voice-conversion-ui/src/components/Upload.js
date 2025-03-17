import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom'; // Import useNavigate
import './Home.css'; // Import the CSS file for button styling

const Upload = () => {
  const [file, setFile] = useState(null);
  const [responseMessage, setResponseMessage] = useState('');
  const [downloadUrl, setDownloadUrl] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate(); // Initialize useNavigate

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    console.log(selectedFile); // Log the selected file
    setFile(selectedFile);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setResponseMessage('Please select a file first.');
      return;
    }

    setIsLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:5000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      console.log(response.data); // Log the response
      setResponseMessage(response.data.output); // Display the output message

      if (response.data.download_url) {
        const downloadUrl = `http://localhost:5000${response.data.download_url}`;
        setDownloadUrl(downloadUrl);
        
        // Get the original URL if available
        const originalUrl = response.data.original_url ? 
          `http://localhost:5000${response.data.original_url}` : '';
        
        // Navigate to the results page with the response data
        navigate('/results', { 
          state: { 
            outputMessage: response.data.output,
            downloadUrl: downloadUrl,
            originalUrl: originalUrl,
            fullPath: response.data.full_path,
            features: response.data.features,
            authenticity: response.data.authenticity,
            gender: response.data.gender
          } 
        });
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      setResponseMessage('Error uploading file. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container"> {/* Centering container */}
      <h2>Upload Voice File</h2>
      <div className="upload-form"> {/* New class for styling */}
        <form onSubmit={handleSubmit}>
          <div className="file-input-container">
            <input 
              type="file" 
              accept=".wav, .mp3, .m4a, .ogg, .flac"  // Allow multiple audio formats
              onChange={handleFileChange} 
              id="file-upload" 
              style={{ display: 'none' }} // Hide the default file input
            />
            <label htmlFor="file-upload" className="button"> {/* Custom label */}
              {file ? file.name : 'Choose File'} {/* Show file name or default text */}
            </label>
          </div>
          <button 
            type="submit" 
            className="button" 
            disabled={isLoading}
          >
            {isLoading ? 'Processing...' : 'Upload'}
          </button>
        </form>
      </div>
      {responseMessage && <p className="response-message">{responseMessage}</p>} {/* Styled response message */}
      {downloadUrl && <a href={downloadUrl} className="download-link">Download Converted Audio</a>} {/* Download link */}
    </div>
  );
};

export default Upload;
