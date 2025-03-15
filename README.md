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

## Deployment on Render
1. Push the project to a GitHub repository.
2. Deploy the backend as a Web Service:
   - Root Directory: `backend`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn server:app`
3. Deploy the frontend as a Static Site:
   - Root Directory: `frontend`
   - Build Command: `npm install && npm run build`
   - Publish Directory: `build`
   - Environment Variable: `REACT_APP_BACKEND_URL=<backend-render-url>`
4. Update CORS in `backend/server.py` with the frontend URL.

## Live Preview
- Frontend: [https://ml-classifier-frontend.onrender.com](https://ml-classifier-frontend.onrender.com)
- Backend: [https://ml-classifier-backend.onrender.com](https://ml-classifier-backend.onrender.com)
