"""
Predictive Modeling Using Machine Learning
========================================
This script loads a dataset, trains multiple supervised machine learning models,
evaluates their accuracy, and visualizes their performance using 
Confusion Matrices and ROC Curves.

Prerequisites:
    pip install numpy pandas scikit-learn matplotlib seaborn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn Modules
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    roc_curve, 
    auc, 
    classification_report
)

def main():
    print("--- Predictive Modeling Application ---")
    
    # 1. Load the Dataset
    # We use the Breast Cancer dataset (binary classification: Malignant or Benign)
    print("\nLoading dataset...")
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features.")
    
    # 2. Split the Data into Training and Testing Sets
    # 80% for training, 20% for testing unseen data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Feature Scaling
    # Many algorithms (like Logistic Regression) perform better when features are on the same scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Initialize Models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    # Dictionaries to store evaluation metrics for plotting later
    roc_data = {}
    confusion_matrices = {}
    
    print("\nTraining and Evaluating Models...")
    
    # 5. Train and Test each model
    for name, model in models.items():
        print(f"\n--- {name} ---")
        
        # Train the model
        model.fit(X_train_scaled, y_train)
        
        # Predict on the test set
        y_pred = model.predict(X_test_scaled)
        
        # Predict probabilities (needed for ROC curve)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        
        # Generate Classification Report (Precision, Recall, F1-Score)
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=data.target_names))
        
        # Store Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices[name] = cm
        
        # Calculate and store ROC Curve data
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}

    # 6. Visualize Performance
    print("\nGenerating visualizations...")
    plot_visualizations(confusion_matrices, roc_data, data.target_names)


def plot_visualizations(confusion_matrices, roc_data, target_names):
    """
    Plots the Confusion Matrices and the ROC Curves for all trained models.
    """
    # Create a figure for Confusion Matrices
    num_models = len(confusion_matrices)
    fig_cm, axes = plt.subplots(1, num_models, figsize=(15, 5))
    fig_cm.suptitle('Confusion Matrices', fontsize=16)
    
    for i, (name, cm) in enumerate(confusion_matrices.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], 
                    xticklabels=target_names, yticklabels=target_names, cbar=False)
        axes[i].set_title(name)
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')
    
    plt.tight_layout()
    
    # Create a figure for ROC Curves
    plt.figure(figsize=(8, 6))
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    
    # Plot ROC curve for each model
    colors = ['blue', 'green', 'red']
    for (name, metrics), color in zip(roc_data.items(), colors):
        plt.plot(metrics['fpr'], metrics['tpr'], color=color, lw=2,
                 label=f"{name} (AUC = {metrics['auc']:.3f})")
    
    # Plot the random guess baseline
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    # Show all plots
    print("Visualizations ready! Displaying windows... (Close the windows to exit the script)")
    plt.show()

if __name__ == "__main__":
    main()