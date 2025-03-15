from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import traceback
import os

app = Flask(__name__)

# Configure CORS to allow requests from the frontend's Render URL
# Replace '<your-frontend-render-url>' with the actual URL after deployment (e.g., https://your-frontend.onrender.com)
CORS(app, resources={r"/predict": {"origins": "*"}})  # Temporarily allow all origins; update later

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        target_column = request.form.get('targetColumn')
        drop_columns = request.form.get('dropColumns', '')
        drop_columns = drop_columns.split(',') if drop_columns else []
        
        df = pd.read_csv(file)
        
        if target_column not in df.columns:
            return jsonify({'error': f"Target column '{target_column}' not found in CSV"}), 400
        
        missing_drop_cols = [col for col in drop_columns if col not in df.columns]
        if missing_drop_cols:
            return jsonify({'error': f"Drop columns not found: {missing_drop_cols}"}), 400
        
        if df[target_column].dtype == 'object':
            df[target_column] = df[target_column].map({'M': 1, 'B': 0}).fillna(df[target_column])
        
        X = df.drop(columns=[target_column] + drop_columns, errors='ignore')
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        imputer = SimpleImputer(strategy='mean')
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
        
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
        models = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Support Vector Machine (SVM)": SVC(),
            "K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB()
        }
        
        results = []
        best_model_name = None
        best_accuracy = 0
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                results.append(f"{name} Accuracy: {accuracy:.4f}")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model_name = name
            except Exception as e:
                results.append(f"{name} Error: {str(e)}")
        
        output = "Model Accuracy Scores:\n" + "\n".join(results)
        output += f"\n\nBest Model: {best_model_name} with Accuracy: {best_accuracy:.4f}"
        
        return jsonify({'results': output})
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Use the PORT environment variable provided by Render
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
