# ML Classifier UI

A web application to compare machine learning models on a CSV dataset using a React frontend and Flask backend.

## Project Structure
- `frontend/`: React app
- `backend/`: Flask app

## Local Setup
1. **Backend**:
   - Navigate to `backend/`
   - Install dependencies: `pip install -r requirements.txt`
   - Run the Flask server: `python server.py`
2. **Frontend**:
   - Navigate to `frontend/`
   - Install dependencies: `npm install`
   - Create a `.env` file with `REACT_APP_BACKEND_URL=http://localhost:5000`
   - Run the React app: `npm start`
