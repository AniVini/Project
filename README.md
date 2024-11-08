Music Genre Classifier using Machine Learning

Overview
This project classifies music into genres using machine learning based on predefined audio features: pitch, timbre, rhythm and harmony. Instead of processing raw audio data, the models use extracted feature vectors, offering a computationally efficient approach. Six different machine learning models were trained and evaluated, with Gradient Boosting showing the highest accuracy and Random Forest as a reliable alternative.

Models Used
The following machine learning models were tested in this project:
- Logistic Regression: A basic linear model adapted for multi-class classification.
- Decision Tree Classifier: Provides interpretable models by building a tree structure based on feature splits.
- Random Forest Classifier: An ensemble of decision trees that improves accuracy and reduces overfitting.
- Support Vector Classifier (SVC): Finds the optimal boundary between classes, effective for high-dimensional features.
- K-Nearest Neighbors (KNN): Classifies based on the majority class among the nearest data points.
- Gradient Boosting Classifier: Sequentially combines weak learners, achieving the highest accuracy in this task.
Key Findings
- Gradient Boosting Classifier emerged as the top-performing model, achieving the highest accuracy and balanced performance across all classes.
- Random Forest Classifier also performed well, making it a strong alternative to Gradient Boosting.
- Other models, while effective, did not achieve the same level of accuracy or consistency as ensemble methods.

Future work could focus on:
- Advanced feature engineering: Extract additional features from the audio data to improve model accuracy.
- Hyperparameter tuning: Further optimize model parameters to boost performance.
- Deep learning: Experiment with neural networks for potentially greater accuracy with larger datasets.
  
Machine learning offers a viable approach for classifying music genres based on key audio features. This project demonstrates the effectiveness of ensemble methods like Gradient Boosting and Random Forest in achieving high accuracy, providing a foundation for more advanced genre classification systems.
