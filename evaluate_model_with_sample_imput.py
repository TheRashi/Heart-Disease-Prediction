import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class HeartDiseasePredictor:
    def __init__(self, data_path="heart.csv"):
        self.data_path = data_path
        self.model = KNeighborsClassifier(n_neighbors=5)
        self.scaler = StandardScaler()
        self.feature_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak',
            'slope', 'ca', 'thal'
        ]

    def prepare_and_train(self):
        """Loads data, scales features, and trains the KNN model."""
        try:
            df = pd.read_csv(self.data_path)
            X = df[self.feature_names]
            y = df["target"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Fit and transform training data
            X_train_scaled = self.scaler.fit_transform(X_train)
            self.model.fit(X_train_scaled, y_train)

            # Quick validation check
            y_pred = self.model.predict(self.scaler.transform(X_test))
            acc = accuracy_score(y_test, y_pred)
            print(f"✅ System Initialized. Model Accuracy: {acc:.2%}")
            
        except FileNotFoundError:
            print("❌ Error: heart.csv not found. Please check the file path.")

    def get_user_prediction(self):
        """Collects user input and returns a prediction."""
        print("\n" + "="*30)
        print(" CLINICAL INPUT TERMINAL ")
        print("="*30)
        
        user_values = []
        for feature in self.feature_names:
            val = float(input(f"➤ Enter {feature.upper()}: "))
            user_values.append(val)

        # Process input
        input_df = pd.DataFrame([user_values], columns=self.feature_names)
        scaled_input = self.scaler.transform(input_df)
        
        prediction = self.model.predict(scaled_input)[0]
        probability = self.model.predict_proba(scaled_input)[0]

        self.display_result(prediction, probability)

    def display_result(self, result, prob):
        """Prints a formatted result based on model output."""
        print("\n" + "-"*30)
        print("DIAGNOSTIC REPORT")
        print("-"*30)
        
        confidence = prob[1] if result == 1 else prob[0]
        
        if result == 1:
            print(f"RESULT: POSITIVE [High Risk]")
            print(f"CONFIDENCE: {confidence:.1%}")
        else:
            print(f"RESULT: NEGATIVE [Low Risk]")
            print(f"CONFIDENCE: {confidence:.1%}")
        print("-"*30)

if __name__ == "__main__":
    # Create instance and run
    predictor = HeartDiseasePredictor("heart.csv")
    predictor.prepare_and_train()
    predictor.get_user_prediction()