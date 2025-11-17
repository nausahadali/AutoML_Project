import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
import joblib
import os
from datetime import datetime

def load_data(file_path):
    """Loads data from a CSV file."""
    print(f"Loading data from {file_path}...")
    return pd.read_csv(file_path)

def infer_problem_type(df, target_column):
    """Infers the problem type (classification or regression) based on the target column."""
    print("Inferring problem type...")
    if df[target_column].dtype == 'object' or df[target_column].nunique() < 20:
        print("--> Inferred problem type: Classification")
        return 'classification'
    else:
        print("--> Inferred problem type: Regression")
        return 'regression'

def get_feature_preprocessor(df, target_column):
    """Creates a preprocessor for numerical and categorical features."""
    print("Defining feature preprocessing steps...")
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df.select_dtypes(include=['object']).columns

    print(f"--> Numeric features: {list(numeric_features)}")
    print(f"--> Categorical features: {list(categorical_features)}")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor

def get_models(problem_type):
    """Returns a dictionary of models based on the problem type."""
    if problem_type == 'classification':
        return {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier()
        }
    else:
        return {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor()
        }

def train_and_evaluate_models(X, y, problem_type, preprocessor):
    """Trains and evaluates a set of models."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = get_models(problem_type)
    best_model = None
    best_score = -1
    best_model_name = ''

    for name, model in models.items():
        print(f"\nTraining {name}...")
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', model)])
        
        pipeline.fit(X_train, y_train)
        
        preds = pipeline.predict(X_test)
        
        if problem_type == 'classification':
            score = accuracy_score(y_test, preds)
            print(f"--> Accuracy: {score:.4f}")
        else:
            score = r2_score(y_test, preds)
            print(f"--> R2 Score: {score:.4f}")

        if score > best_score:
            best_score = score
            best_model = pipeline
            best_model_name = name
            
    print(f"\nBest model: {best_model_name} with a score of {best_score:.4f}")
    return best_model, best_model_name, best_score

def save_model(model, model_name, dataset_name):
    """Saves the trained model to a file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{dataset_name}_{model_name.replace(' ', '_')}_{timestamp}.joblib"
    save_path = os.path.join('trained_models', filename)
    
    print(f"Saving model to {save_path}...")
    joblib.dump(model, save_path)
    print("--> Model saved successfully.")
    return save_path

def main():
    """Main function to run the AutoML pipeline."""
    parser = argparse.ArgumentParser(description="A simple AutoML command-line tool.")
    parser.add_argument("dataset_path", help="Path to the CSV dataset.")
    parser.add_argument("target_column", help="Name of the target column to predict.")
    args = parser.parse_args()

    try:
        # Load data
        df = load_data(args.dataset_path)

        # Infer problem type
        problem_type = infer_problem_type(df, args.target_column)

        # Separate features and target
        X = df.drop(columns=[args.target_column])
        y = df[args.target_column]

        # Create preprocessor
        preprocessor = get_feature_preprocessor(X, args.target_column)

        # Train and evaluate models
        best_model, best_model_name, best_score = train_and_evaluate_models(X, y, problem_type, preprocessor)

        # Save the best model
        dataset_name = os.path.splitext(os.path.basename(args.dataset_path))[0]
        saved_model_path = save_model(best_model, best_model_name, dataset_name)

        print("\n--- AutoML Run Summary ---")
        print(f"Dataset: {args.dataset_path}")
        print(f"Target Column: {args.target_column}")
        print(f"Problem Type: {problem_type}")
        print(f"Best Performing Model: {best_model_name}")
        print(f"Score: {best_score:.4f}")
        print(f"Saved Model Path: {saved_model_path}")
        print("--------------------------")

    except FileNotFoundError:
        print(f"Error: The file '{args.dataset_path}' was not found.")
    except KeyError:
        print(f"Error: The target column '{args.target_column}' was not found in the dataset.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
