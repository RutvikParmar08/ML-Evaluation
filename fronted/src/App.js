import axios from 'axios';
import React, { useEffect, useState } from 'react';
import Dropzone from 'react-dropzone';

function App() {
  const [file, setFile] = useState(null);
  const [targetColumn, setTargetColumn] = useState('');
  const [dropColumns, setDropColumns] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [darkMode, setDarkMode] = useState(false);

  // Check for user's preferred color scheme on initial load
  useEffect(() => {
    const prefersDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
    setDarkMode(prefersDarkMode);
  }, []);

  // Toggle dark mode
  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
  };

  const onDrop = (acceptedFiles) => {
    setFile(acceptedFiles[0]);
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a CSV file');
      return;
    }

    setLoading(true);
    setError('');
    setResults(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('targetColumn', targetColumn);
    formData.append('dropColumns', dropColumns);

    try {
      const response = await axios.post('http://localhost:5000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setResults(response.data.results);
    } catch (err) {
      setError(err.response?.data?.error || 'Network Error: Could not connect to the server');
    } finally {
      setLoading(false);
    }
  };

  const parseResults = (resultsText) => {
    if (!resultsText) return { accuracies: [], bestModel: '', bestAccuracy: 0 };

    const lines = resultsText.split('\n');
    const accuracies = [];
    let bestModel = '';
    let bestAccuracy = 0;

    lines.forEach((line) => {
      line = line.trim();
      if (line.includes('Accuracy:') && !line.includes('Best Model:')) {
        const [namePart, accuracyPart] = line.split(' Accuracy: ');
        const name = namePart.trim();
        const accuracy = parseFloat(accuracyPart);
        if (!isNaN(accuracy)) {
          accuracies.push({ name, accuracy });
        }
      } else if (line.includes('Error:') && !line.includes('Best Model:')) {
        const [namePart, errorPart] = line.split(' Error: ');
        const name = namePart.trim();
        accuracies.push({ name, accuracy: 'Error', error: errorPart });
      } else if (line.startsWith('Best Model:')) {
        const parts = line.replace('Best Model:', '').split(' with Accuracy: ');
        bestModel = parts[0].trim();
        bestAccuracy = parseFloat(parts[1]);
      }
    });

    // Fallback to calculate best model if not explicitly provided
    if (!bestModel && accuracies.length > 0) {
      const validAccuracies = accuracies.filter(a => a.accuracy !== 'Error');
      if (validAccuracies.length > 0) {
        const best = validAccuracies.reduce((prev, curr) =>
          curr.accuracy > prev.accuracy ? curr : prev
        );
        bestModel = best.name;
        bestAccuracy = best.accuracy;
      }
    }

    return { accuracies, bestModel, bestAccuracy };
  };

  const { accuracies, bestModel, bestAccuracy } = parseResults(results);

  // CSS styles with dark mode support
  const styles = `
    /* Base styles */
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
      background-color: ${darkMode ? '#121212' : '#f3f4f6'};
      color: ${darkMode ? '#e5e7eb' : '#1f2937'};
      transition: background-color 0.3s, color 0.3s;
    }
    
    /* Container and layout */
    .container {
      min-height: 100vh;
      padding: 1rem;
      background-color: ${darkMode ? '#121212' : '#f3f4f6'};
      transition: background-color 0.3s;
    }
    
    .card {
      max-width: 1024px;
      margin: 0 auto;
      background-color: ${darkMode ? '#1f2937' : 'white'};
      border-radius: 0.5rem;
      box-shadow: 0 1px 3px 0 rgba(0, 0, 0, ${darkMode ? '0.3' : '0.1'}), 0 1px 2px 0 rgba(0, 0, 0, ${darkMode ? '0.3' : '0.06'});
      overflow: hidden;
      transition: background-color 0.3s, box-shadow 0.3s;
    }
    
    .card-header {
      padding: 1.5rem;
      border-bottom: 1px solid ${darkMode ? '#374151' : '#e5e7eb'};
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      transition: border-color 0.3s;
    }
    
    .header-content {
      flex: 1;
      min-width: 200px;
    }
    
    .card-body {
      padding: 1.5rem;
    }
    
    /* Typography */
    h1 {
      font-size: 1.875rem;
      font-weight: bold;
      margin: 0;
      color: ${darkMode ? '#f9fafb' : '#1f2937'};
      transition: color 0.3s;
    }
    
    h2 {
      font-size: 1.25rem;
      font-weight: 600;
      margin-top: 0;
      margin-bottom: 1rem;
      color: ${darkMode ? '#f9fafb' : '#1f2937'};
      transition: color 0.3s;
    }
    
    p {
      margin-top: 0.5rem;
      color: ${darkMode ? '#d1d5db' : '#4b5563'};
      transition: color 0.3s;
    }
    
    /* Form elements */
    .form-group {
      margin-bottom: 1.5rem;
    }
    
    label {
      display: block;
      font-size: 0.875rem;
      font-weight: 500;
      margin-bottom: 0.5rem;
      color: ${darkMode ? '#d1d5db' : '#374151'};
      transition: color 0.3s;
    }
    
    input[type="text"] {
      width: 100%;
      padding: 0.75rem;
      border: 1px solid ${darkMode ? '#4b5563' : '#d1d5db'};
      border-radius: 0.375rem;
      box-shadow: 0 1px 2px 0 rgba(0, 0, 0, ${darkMode ? '0.2' : '0.05'});
      font-size: 1rem;
      background-color: ${darkMode ? '#374151' : 'white'};
      color: ${darkMode ? '#f9fafb' : 'inherit'};
      transition: border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out, background-color 0.3s, color 0.3s;
    }
    
    input[type="text"]:focus {
      border-color: ${darkMode ? '#60a5fa' : '#3b82f6'};
      outline: none;
      box-shadow: 0 0 0 3px rgba(${darkMode ? '96, 165, 250' : '59, 130, 246'}, 0.25);
    }
    
    /* Dropzone */
    .dropzone {
      border: 2px dashed ${darkMode ? '#4b5563' : '#d1d5db'};
      border-radius: 0.375rem;
      padding: 2rem 1rem;
      text-align: center;
      cursor: pointer;
      background-color: ${darkMode ? '#283141' : '#f9fafb'};
      transition: all 0.2s ease, background-color 0.3s, border-color 0.3s;
    }
    
    .dropzone:hover {
      background-color: ${darkMode ? '#374151' : '#f3f4f6'};
      border-color: ${darkMode ? '#6b7280' : '#9ca3af'};
    }
    
    .dropzone.active {
      border-color: ${darkMode ? '#34d399' : '#10b981'};
      background-color: ${darkMode ? '#064e3b' : '#ecfdf5'};
    }
    
    .dropzone-icon {
      width: 3rem;
      height: 3rem;
      margin: 0 auto 0.5rem;
      color: ${darkMode ? '#6b7280' : '#9ca3af'};
      transition: color 0.3s;
    }
    
    .dropzone-icon.success {
      color: ${darkMode ? '#34d399' : '#10b981'};
    }
    
    .dropzone-text {
      font-weight: 500;
      margin-bottom: 0.25rem;
      color: ${darkMode ? '#d1d5db' : 'inherit'};
      transition: color 0.3s;
    }
    
    .dropzone-subtext {
      font-size: 0.75rem;
      color: ${darkMode ? '#9ca3af' : '#6b7280'};
      transition: color 0.3s;
    }
    
    /* Button */
    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 0.75rem 1.5rem;
      font-size: 0.875rem;
      font-weight: 500;
      border-radius: 0.375rem;
      border: none;
      cursor: pointer;
      transition: background-color 0.15s ease-in-out;
    }
    
    .btn-primary {
      background-color: ${darkMode ? '#3b82f6' : '#2563eb'};
      color: white;
    }
    
    .btn-primary:hover {
      background-color: ${darkMode ? '#2563eb' : '#1d4ed8'};
    }
    
    .btn-primary:disabled {
      opacity: 0.65;
      cursor: not-allowed;
    }
    
    /* Spinner */
    .spinner {
      animation: spin 1s linear infinite;
      margin-right: 0.5rem;
      width: 1rem;
      height: 1rem;
    }
    
    @keyframes spin {
      from {
        transform: rotate(0deg);
      }
      to {
        transform: rotate(360deg);
      }
    }
    
    /* Error message */
    .error {
      margin-top: 1rem;
      padding: 0.75rem 1rem;
      background-color: ${darkMode ? '#7f1d1d' : '#fee2e2'};
      border-left: 4px solid ${darkMode ? '#ef4444' : '#ef4444'};
      border-radius: 0.25rem;
      color: ${darkMode ? '#fca5a5' : '#b91c1c'};
      display: flex;
      align-items: flex-start;
      transition: background-color 0.3s, color 0.3s;
    }
    
    .error-icon {
      flex-shrink: 0;
      margin-right: 0.75rem;
      width: 1.25rem;
      height: 1.25rem;
    }
    
    /* Results section */
    .results-section {
      margin-top: 2rem;
      padding-top: 1.5rem;
      border-top: 1px solid ${darkMode ? '#374151' : '#e5e7eb'};
      transition: border-color 0.3s;
    }
    
    .table-container {
      overflow-x: auto;
      border-radius: 0.5rem;
      border: 1px solid ${darkMode ? '#374151' : '#e5e7eb'};
      margin-bottom: 1.5rem;
      transition: border-color 0.3s;
    }
    
    table {
      width: 100%;
      border-collapse: collapse;
    }
    
    th {
      background-color: ${darkMode ? '#283141' : '#f3f4f6'};
      padding: 0.75rem 1rem;
      text-align: left;
      font-size: 0.75rem;
      font-weight: 600;
      color: ${darkMode ? '#9ca3af' : '#6b7280'};
      text-transform: uppercase;
      letter-spacing: 0.05em;
      border-bottom: 1px solid ${darkMode ? '#374151' : '#e5e7eb'};
      transition: background-color 0.3s, color 0.3s, border-color 0.3s;
    }
    
    td {
      padding: 0.75rem 1rem;
      font-size: 0.875rem;
      border-bottom: 1px solid ${darkMode ? '#374151' : '#e5e7eb'};
      transition: border-color 0.3s;
    }
    
    tr:last-child td {
      border-bottom: none;
    }
    
    tr.best-model {
      background-color: ${darkMode ? '#1e3a8a' : '#eff6ff'};
      transition: background-color 0.3s;
    }
    
    /* Progress bar */
    .progress-container {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    
    .progress-bar {
      flex-grow: 1;
      height: 0.5rem;
      background-color: ${darkMode ? '#374151' : '#e5e7eb'};
      border-radius: 9999px;
      overflow: hidden;
      max-width: 200px;
      transition: background-color 0.3s;
    }
    
    .progress-fill {
      height: 100%;
      background-color: ${darkMode ? '#3b82f6' : '#2563eb'};
      border-radius: 9999px;
      transition: background-color 0.3s;
    }
    
    .progress-text {
      font-weight: 500;
      min-width: 3.5rem;
      text-align: right;
    }
    
    /* Best model summary */
    .best-model-summary {
      margin-top: 1.5rem;
      padding: 1rem;
      background-color: ${darkMode ? '#1e3a8a' : '#eff6ff'};
      border: 1px solid ${darkMode ? '#3b82f6' : '#bfdbfe'};
      border-radius: 0.5rem;
      display: flex;
      align-items: center;
      transition: background-color 0.3s, border-color 0.3s;
    }
    
    .summary-icon-container {
      flex-shrink: 0;
      width: 2.5rem;
      height: 2.5rem;
      border-radius: 9999px;
      background-color: ${darkMode ? '#2563eb' : '#dbeafe'};
      display: flex;
      align-items: center;
      justify-content: center;
      margin-right: 1rem;
      transition: background-color 0.3s;
    }
    
    .summary-icon {
      width: 1.5rem;
      height: 1.5rem;
      color: ${darkMode ? '#bfdbfe' : '#3b82f6'};
      transition: color 0.3s;
    }
    
    .summary-text h3 {
      margin: 0;
      font-size: 1rem;
      font-weight: 600;
      color: ${darkMode ? '#bfdbfe' : '#1e40af'};
      transition: color 0.3s;
    }
    
    .summary-text p {
      margin: 0.25rem 0 0 0;
      color: ${darkMode ? '#93c5fd' : '#1e40af'};
      transition: color 0.3s;
    }
    
    /* Responsive grid */
    .grid {
      display: grid;
      gap: 1rem;
    }
    
    @media (min-width: 768px) {
      .grid-cols-2 {
        grid-template-columns: repeat(2, 1fr);
      }
    }
    
    /* Dark mode toggle */
    .dark-mode-toggle {
      position: relative;
      display: inline-block;
      width: 60px;
      height: 30px;
      margin-left: 1rem;
    }
    
    .dark-mode-toggle input {
      opacity: 0;
      width: 0;
      height: 0;
    }
    
    .slider {
      position: absolute;
      cursor: pointer;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background-color: ${darkMode ? '#4b5563' : '#e5e7eb'};
      transition: .4s;
      border-radius: 34px;
    }
    
    .slider:before {
      position: absolute;
      content: "";
      height: 22px;
      width: 22px;
      left: 4px;
      bottom: 4px;
      background-color: white;
      transition: .4s;
      border-radius: 50%;
    }
    
    input:checked + .slider {
      background-color: #3b82f6;
    }
    
    input:checked + .slider:before {
      transform: translateX(30px);
    }
    
    .toggle-icon {
      position: absolute;
      top: 6px;
      font-size: 14px;
    }
    
    .moon-icon {
      right: 8px;
      color: #f9fafb;
    }
    
    .sun-icon {
      left: 8px;
      color: #f59e0b;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 640px) {
      .card-header {
        flex-direction: column;
        align-items: flex-start;
      }
      
      .dark-mode-switcher {
        margin-top: 1rem;
        display: flex;
        align-items: center;
      }
      
      .dark-mode-label {
        margin-right: 0.5rem;
      }
      
      h1 {
        font-size: 1.5rem;
      }
      
      .dropzone {
        padding: 1.5rem 1rem;
      }
      
      .grid-cols-2 {
        grid-template-columns: 1fr;
      }
      
      .btn-primary {
        width: 100%;
      }
    }
  `;

  return (
    <>
      <style>{styles}</style>
      <div className="container">
        <div className="card">
          <header className="card-header">
            <div className="header-content">
              <h1>Machine Learning Classifier</h1>
              <p>Upload a CSV file to compare multiple ML models</p>
            </div>
            <div className="dark-mode-switcher">
              <span className="dark-mode-label">
                {darkMode ? 'Night Mode' : 'Day Mode'}
              </span>
              <label className="dark-mode-toggle">
                <input 
                  type="checkbox" 
                  checked={darkMode} 
                  onChange={toggleDarkMode}
                />
                <span className="slider">
                  <span className="toggle-icon sun-icon">☀️</span>
                  <span className="toggle-icon moon-icon">🌙</span>
                </span>
              </label>
            </div>
          </header>

          <main className="card-body">
            <form onSubmit={handleSubmit}>
              <div className="form-group">
                <label>Dataset</label>
                <Dropzone
                  onDrop={onDrop}
                  accept={{ 'text/csv': ['.csv'] }}
                  multiple={false}
                >
                  {({ getRootProps, getInputProps, isDragActive }) => (
                    <div 
                      {...getRootProps()} 
                      className={`dropzone ${isDragActive ? 'active' : ''} ${file ? 'active' : ''}`}
                    >
                      <input {...getInputProps()} />
                      {file ? (
                        <>
                          <svg className="dropzone-icon success" viewBox="0 0 20 20" fill="currentColor">
                            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                          </svg>
                          <p className="dropzone-text">{file.name}</p>
                          <p className="dropzone-subtext">Click or drag to replace</p>
                        </>
                      ) : (
                        <>
                          <svg className="dropzone-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16.5 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                          </svg>
                          <p className="dropzone-text">Drop CSV file here</p>
                          <p className="dropzone-subtext">or click to browse files</p>
                        </>
                      )}
                    </div>
                  )}
                </Dropzone>
              </div>

              <div className="grid grid-cols-2">
                <div className="form-group">
                  <label>Target Column</label>
                  <input
                    type="text"
                    value={targetColumn}
                    onChange={(e) => setTargetColumn(e.target.value)}
                    placeholder="e.g., diagnosis"
                  />
                </div>

                <div className="form-group">
                  <label>Columns to Drop (comma-separated)</label>
                  <input
                    type="text"
                    value={dropColumns}
                    onChange={(e) => setDropColumns(e.target.value)}
                    placeholder="e.g., id"
                  />
                </div>
              </div>

              <button
                type="submit"
                disabled={loading}
                className="btn btn-primary"
              >
                {loading ? (
                  <>
                    <svg className="spinner" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Processing...
                  </>
                ) : (
                  'Run Models'
                )}
              </button>

              {error && (
                <div className="error">
                  <svg className="error-icon" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                  <span>{error}</span>
                </div>
              )}
            </form>

            {results && (
              <div className="results-section">
                <h2>Model Performance</h2>
                <div className="table-container">
                  <table>
                    <thead>
                      <tr>
                        <th>Model</th>
                        <th>Accuracy</th>
                      </tr>
                    </thead>
                    <tbody>
                      {accuracies.map((model, index) => (
                        <tr
                          key={index}
                          className={model.name === bestModel ? 'best-model' : ''}
                        >
                          <td>
                            {model.name === bestModel && (
                              <svg style={{ display: 'inline-block', width: '1.25rem', height: '1.25rem', marginRight: '0.5rem', verticalAlign: 'text-bottom', color: darkMode ? '#3b82f6' : '#3b82f6' }} fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                              </svg>
                            )}
                            {model.name}
                          </td>
                          <td>
                            {model.accuracy === 'Error' ? (
                              <span style={{ color: darkMode ? '#f87171' : '#dc2626' }}>Error: {model.error}</span>
                            ) : (
                              <div className="progress-container">
                                <div className="progress-bar">
                                  <div 
                                    className="progress-fill" 
                                    style={{ width: `${model.accuracy * 100}%` }}
                                  ></div>
                                </div>
                                <span className="progress-text">{(model.accuracy * 100).toFixed(2)}%</span>
                              </div>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                
                {bestModel && (
                  <div className="best-model-summary">
                    <div className="summary-icon-container">
                      <svg className="summary-icon" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                    </div>
                    <div className="summary-text">
                      <h3>Best Model: {bestModel}</h3>
                      <p>Accuracy: <strong>{(bestAccuracy * 100).toFixed(2)}%</strong></p>
                    </div>
                  </div>
                )}
              </div>
            )}
          </main>
        </div>
      </div>
    </>
  );
}

export default App;