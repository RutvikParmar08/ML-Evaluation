# # from flask import Flask, request, jsonify
# # from flask_cors import CORS
# # import pandas as pd
# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.tree import DecisionTreeClassifier
# # from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# # from sklearn.svm import SVC
# # from sklearn.neighbors import KNeighborsClassifier
# # from sklearn.naive_bayes import GaussianNB
# # from sklearn.metrics import accuracy_score
# # from sklearn.impute import SimpleImputer
# # import traceback

# # app = Flask(__name__)
# # CORS(app)  # Enable CORS

# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     try:
# #         if 'file' not in request.files:
# #             return jsonify({'error': 'No file part in the request'}), 400
        
# #         file = request.files['file']
# #         if file.filename == '':
# #             return jsonify({'error': 'No file selected'}), 400
        
# #         target_column = request.form.get('targetColumn')
# #         drop_columns = request.form.get('dropColumns', '')
# #         drop_columns = drop_columns.split(',') if drop_columns else []
        
# #         df = pd.read_csv(file)
        
# #         if target_column not in df.columns:
# #             return jsonify({'error': f"Target column '{target_column}' not found in CSV"}), 400
        
# #         missing_drop_cols = [col for col in drop_columns if col not in df.columns]
# #         if missing_drop_cols:
# #             return jsonify({'error': f"Drop columns not found: {missing_drop_cols}"}), 400
        
# #         if df[target_column].dtype == 'object':
# #             df[target_column] = df[target_column].map({'M': 1, 'B': 0}).fillna(df[target_column])
        
# #         X = df.drop(columns=[target_column] + drop_columns, errors='ignore')
# #         y = df[target_column]
        
# #         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
# #         imputer = SimpleImputer(strategy='mean')
# #         X_train = imputer.fit_transform(X_train)
# #         X_test = imputer.transform(X_test)
        
# #         sc = StandardScaler()
# #         X_train = sc.fit_transform(X_train)
# #         X_test = sc.transform(X_test)
        
# #         models = {
# #             "Logistic Regression": LogisticRegression(),
# #             "Decision Tree": DecisionTreeClassifier(),
# #             "Random Forest": RandomForestClassifier(),
# #             "Gradient Boosting": GradientBoostingClassifier(),
# #             "Support Vector Machine (SVM)": SVC(),
# #             "K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
# #             "Naive Bayes": GaussianNB()
# #         }
        
# #         results = []
# #         best_model_name = None
# #         best_accuracy = 0
        
# #         for name, model in models.items():
# #             try:
# #                 model.fit(X_train, y_train)
# #                 y_pred = model.predict(X_test)
# #                 accuracy = accuracy_score(y_test, y_pred)
# #                 results.append(f"{name} Accuracy: {accuracy:.4f}")
                
# #                 if accuracy > best_accuracy:
# #                     best_accuracy = accuracy
# #                     best_model_name = name
# #             except Exception as e:
# #                 results.append(f"{name} Error: {str(e)}")
        
# #         output = "Model Accuracy Scores:\n" + "\n".join(results)
# #         output += f"\n\nBest Model: {best_model_name} with Accuracy: {best_accuracy:.4f}"
        
# #         return jsonify({'results': output})
    
# #     except Exception as e:
# #         traceback.print_exc()
# #         return jsonify({'error': str(e)}), 500

# # if __name__ == '__main__':
# #     app.run(debug=True, port=5000)
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression, LinearRegression
# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
# from sklearn.svm import SVC, SVR
# from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score, r2_score
# from sklearn.impute import SimpleImputer
# import traceback

# app = Flask(__name__)
# CORS(app)  # Enable CORS

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         print("Received Request") 
#         if 'file' not in request.files:
#             return jsonify({'error': 'No file part in the request'}), 400
        
#         file = request.files['file']
#         if file.filename == '':
#             return jsonify({'error': 'No file selected'}), 400
        
#         target_column = request.form.get('targetColumn')
#         print("Target Column:", target_column)

#         drop_columns = request.form.get('dropColumns', '')
#         drop_columns = drop_columns.split(',') if drop_columns else []

#         print("Drop Columns:", drop_columns)
#         print("Reading CSV file...")
#         df = pd.read_csv(file)
#         print("CSV file loaded!")
        
#         if target_column not in df.columns:
#             return jsonify({'error': f"Target column '{target_column}' not found in CSV"}), 400
        
#         missing_drop_cols = [col for col in drop_columns if col not in df.columns]
#         if missing_drop_cols:
#             return jsonify({'error': f"Drop columns not found: {missing_drop_cols}"}), 400
        
#         # Prepare target variable
#         if df[target_column].dtype == 'object':
#             df[target_column] = df[target_column].map({'M': 1, 'B': 0}).fillna(df[target_column])

#         X = df.drop(columns=[target_column] + drop_columns, errors='ignore')
#         y = df[target_column]

#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         # Handle missing values
#         imputer = SimpleImputer(strategy='mean')
#         X_train = imputer.fit_transform(X_train)
#         X_test = imputer.transform(X_test)

#         # Scale features
#         sc = StandardScaler()
#         X_train = sc.fit_transform(X_train)
#         X_test = sc.transform(X_test)

#         # Detect classification or regression
#         if y.dtype in ['float64', 'int64'] and len(y.unique()) > 10:  # More than 10 unique values → Regression
#             problem_type = "Regression"
#             models = {
#                 "Linear Regression": LinearRegression(),
#                 "Decision Tree Regressor": DecisionTreeRegressor(),
#                 "Random Forest Regressor": RandomForestRegressor(),
#                 "Gradient Boosting Regressor": GradientBoostingRegressor(),
#                 "Support Vector Regressor (SVR)": SVR(),
#                 "K-Nearest Neighbors Regressor (KNN)": KNeighborsRegressor()
#             }
#             metric_function = r2_score
#             metric_name = "R² Score"
#         else:  # Less than 10 unique values → Classification
#             problem_type = "Classification"
#             models = {
#                 "Logistic Regression": LogisticRegression(),
#                 "Decision Tree": DecisionTreeClassifier(),
#                 "Random Forest": RandomForestClassifier(),
#                 "Gradient Boosting": GradientBoostingClassifier(),
#                 "Support Vector Machine (SVM)": SVC(),
#                 "K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
#                 "Naive Bayes": GaussianNB()
#             }
#             metric_function = accuracy_score
#             metric_name = "Accuracy"

#         results = []
#         best_model_name = None
#         best_score = float('-inf')

#         model_results = {}
#         print("...................")
#         for name, model in models.items():
#             try:
#                 model.fit(X_train, y_train)
#                 y_pred = model.predict(X_test)

#                 score = metric_function(y_test, y_pred)

#                 model_results[name] = round(score, 4)

#                 if score > best_score:
#                     best_score = score
#                     best_model_name = name
#             except Exception as e:
#                 model_results[name] = f"Error: {str(e)}"

#         return jsonify({
#             'problem_type': problem_type,
#             'model_results': model_results,
#             'best_model': best_model_name,
#             'best_score': round(best_score, 4)
#         })

#     except Exception as e:
#         traceback.print_exc()
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression, LinearRegression
# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
# from sklearn.svm import SVC, SVR
# from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
# from sklearn.impute import SimpleImputer
# import traceback

# app = Flask(__name__)
# CORS(app)  # Enable CORS

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         if 'file' not in request.files:
#             return jsonify({'error': 'No file part in the request'}), 400
        
#         file = request.files['file']
#         if file.filename == '':
#             return jsonify({'error': 'No file selected'}), 400
        
#         target_column = request.form.get('targetColumn')
#         drop_columns = request.form.get('dropColumns', '')
#         drop_columns = drop_columns.split(',') if drop_columns else []
        
#         df = pd.read_csv(file)
        
#         if target_column not in df.columns:
#             return jsonify({'error': f"Target column '{target_column}' not found in CSV"}), 400
        
#         missing_drop_cols = [col for col in drop_columns if col not in df.columns]
#         if missing_drop_cols:
#             return jsonify({'error': f"Drop columns not found: {missing_drop_cols}"}), 400
        
#         # Prepare target variable
#         if df[target_column].dtype == 'object':
#             df[target_column] = df[target_column].map({'M': 1, 'B': 0}).fillna(df[target_column])

#         X = df.drop(columns=[target_column] + drop_columns, errors='ignore')
#         y = df[target_column]

#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         # Handle missing values
#         imputer = SimpleImputer(strategy='mean')
#         X_train = imputer.fit_transform(X_train)
#         X_test = imputer.transform(X_test)

#         # Scale features
#         sc = StandardScaler()
#         X_train = sc.fit_transform(X_train)
#         X_test = sc.transform(X_test)

#         # Detect classification or regression
#         if y.dtype in ['float64', 'int64'] and len(y.unique()) > 10:  # More than 10 unique values → Regression
#             problem_type = "Regression"
#             models = {
#                 "Linear Regression": LinearRegression(),
#                 "Decision Tree Regressor": DecisionTreeRegressor(),
#                 "Random Forest Regressor": RandomForestRegressor(),
#                 "Gradient Boosting Regressor": GradientBoostingRegressor(),
#                 "Support Vector Regressor (SVR)": SVR(),
#                 "K-Nearest Neighbors Regressor (KNN)": KNeighborsRegressor()
#             }
#         else:  # Less than 10 unique values → Classification
#             problem_type = "Classification"
#             models = {
#                 "Logistic Regression": LogisticRegression(),
#                 "Decision Tree": DecisionTreeClassifier(),
#                 "Random Forest": RandomForestClassifier(),
#                 "Gradient Boosting": GradientBoostingClassifier(),
#                 "Support Vector Machine (SVM)": SVC(),
#                 "K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
#                 "Naive Bayes": GaussianNB()
#             }

#         results = []
#         best_model_name = None
#         best_score = 0

#         for name, model in models.items():
#             try:
#                 model.fit(X_train, y_train)
#                 y_pred = model.predict(X_test)

#                 if problem_type == "Classification":
#                     score = accuracy_score(y_test, y_pred)
#                     metric = "Accuracy"
#                 else:
#                     score = r2_score(y_test, y_pred)  # R-squared for regression
#                     metric = "R² Score"

#                 results.append(f"{name} {metric}: {score:.4f}")

#                 if score > best_score:
#                     best_score = score
#                     best_model_name = name
#             except Exception as e:
#                 results.append(f"{name} Error: {str(e)}")

#         output = f"{problem_type} Model Scores:\n" + "\n".join(results)
#         output += f"\n\nBest Model: {best_model_name} with {metric}: {best_score:.4f}"

#         return jsonify({'results': output})

#     except Exception as e:
#         traceback.print_exc()
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS

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
        
        # Separate features (X) and target variable (y)
        X = df.drop(columns=[target_column] + drop_columns, errors='ignore')
        y = df[target_column]

        # Encode categorical target variable (if applicable)
        if y.dtype == 'object':
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

        # Identify categorical columns
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

        # Encode categorical features
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)  # One-hot encoding
        
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        # Scale numerical features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Detect classification or regression
        if y.dtype in ['float64', 'int64'] and len(set(y)) > 10:  # More than 10 unique values → Regression
            problem_type = "Regression"
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "Support Vector Regressor (SVR)": SVR(),
                "K-Nearest Neighbors Regressor (KNN)": KNeighborsRegressor()
            }
        else:  # Less than 10 unique values → Classification
            problem_type = "Classification"
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
        best_score = 0

        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                if problem_type == "Classification":
                    score = accuracy_score(y_test, y_pred)
                    metric = "Accuracy"
                else:
                    score = r2_score(y_test, y_pred)  # R² score for regression
                    metric = "R² Score"

                results.append(f"{name} {metric}: {score:.4f}")

                if score > best_score:
                    best_score = score
                    best_model_name = name
            except Exception as e:
                results.append(f"{name} Error: {str(e)}")

        output = f"{problem_type} Model Scores:\n" + "\n".join(results)
        output += f"\n\nBest Model: {best_model_name} with {metric}: {best_score:.4f}"

        return jsonify({'results': output})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
