# Comparative Analysis of Machine Learning and Deep Learning Algorithms for Credit Card Fraud Detection on 2nd Dataset (Simulated credit card data) 
![image](https://github.com/Khadija-khanom/credit_card_fraud_detection2/assets/138976722/a4cfb6d1-532a-47bb-b4e6-101ce6f3b17e)

This is a simulated credit card transaction dataset containing legitimate and fraud transactions from the duration 1st Jan 2019 - 31st Dec 2020. It covers credit cards of 1000 customers doing transactions with a pool of 800 merchants.
Attribute	Representation

**Unnamed:**0	 An index column, possibly representing a unique identifier for each row

**trans_date_trans_time:** The date and time of the transaction

**cc_num:** 	The credit card number used for the transaction

**merchant:**	The name of the merchant where the transaction took place

**category:**	The category of the transaction (e.g., grocery, travel, etc.)
amt	The amount of the transaction

**first:**	The first name of the cardholder

**last:**	The last name of the cardholder

**gender:**	The gender of the cardholder

**street:**	The street address of the cardholder

**city:**	The city of the cardholder

**state:**	The state of the cardholder

**zip:**	The zip code of the cardholder

**lat:**	The latitude of the cardholder's location

**long:**	The longitude of the cardholder's location

**city_pop:**	The population of the city where the cardholder resides

**job:**	The occupation or job title of the cardholder

**dob:**	The date of birth of the cardholder

**trans_num:**	A unique identifier for each transaction

**unix_time:** 	The transaction time in UNIX timestamp format

**merch_lat:**	The latitude of the merchant's location

**Objective:**
The primary objective of this study is to build and evaluate predictive models capable of identifying fraudulent credit card transactions with high accuracy, while minimizing false positives and false negatives. This involves the application of various machine learning and deep learning techniques to effectively handle the imbalanced nature of the dataset and capture intricate patterns associated with fraudulent activities.

Table of Contents
=================




## Data Visualization

**plotting amt Features:** 
![image](https://github.com/Khadija-khanom/credit_card_fraud_detection2/assets/138976722/054b39a1-1307-4c9e-8770-b48e27421fa0)


**gender-fraud distribution:**

![image](https://github.com/Khadija-khanom/credit_card_fraud_detection2/assets/138976722/d2845367-6cd9-436b-9c5d-7f5e9b9ba8c4)


**The age-fraud distribution:**

![image](https://github.com/Khadija-khanom/credit_card_fraud_detection2/assets/138976722/ec13bb74-5b1c-4768-bb15-e5b99f59318d)


**the number of fraudulent transactions in each category:**

![image](https://github.com/Khadija-khanom/credit_card_fraud_detection2/assets/138976722/285eefca-7eb7-41f4-ac18-9d2db12d294f)


**The correlations between the columns**

![image](https://github.com/Khadija-khanom/credit_card_fraud_detection2/assets/138976722/90110616-c06e-4214-bc48-9998b591ab75)




## Implementation of Machine learning Models
### Learning Algorithm for Machine Learning Models:

**Load and preprocess the dataset:**

- Load dataset 'processed.csv'.

- Handle any missing values and standardize features.

**Balance class distribution using SMOTE:** Apply SMOTE to the dataset to address class imbalance.

**Shuffle the data:** Randomly shuffle the data for randomness.

**Split data into training and testing sets:** Split data into training and testing sets (e.g., 80% training and 20% testing).

**Build and train machine learning models:**

For each machine learning model (KNN, Logistic Regression, Decision Tree, Random Forest):

- Create the model instance.
- Train the model using the training data (X_train, y_train).

**Model Evaluation:**

For each model:

- Predict the target values (y_pred) using the testing data (X_test).
- Calculate accuracy using accuracy_score(y_test, y_pred).

**Generate evaluation metrics:**

For each model:

- Generate a classification report (classification_report) with precision, recall, and F1-score.

- Generate a confusion matrix (confusion_matrix) with true positives, true negatives, false positives, and false negatives.

**Print results:** Print accuracy, classification report, and confusion matrix for each model.

This algorithm outlines the steps involved in building, training, and evaluating the machine learning models (KNN, Logistic Regression, Decision Tree, Random Forest) for credit card fraud detection.

## Evaluating the Performance of Machine Learning Models
###  K-Nearest Neighbors model 

<img width="340" alt="image" src="https://github.com/Khadija-khanom/credit_card_fraud_detection2/assets/138976722/dc915a8e-7601-43db-988e-5c4c07fe5a1a">

The classification report and confusion matrix provide insights into the performance of the K-Nearest Neighbors (KNN) model on the dataset:

1. **Precision, Recall, and F1-Score:** For both classes (0 and 1), precision, recall, and F1-score are all around 0.99. This indicates that the KNN model is performing very well for both classes, with a high ability to identify true positives (recall) and minimize false positives (precision).

2. **Support:** The support indicates the number of instances of each class in the dataset. Both classes have similar support, suggesting a balanced dataset.

3. **Accuracy:** The overall accuracy of the KNN model is approximately 99%, indicating that the model correctly predicted the class of about 99% of the instances.

4. **Macro Avg and Weighted Avg:** Both macro and weighted averages for precision, recall, and F1-score are around 0.99, aligning with the performance metrics of individual classes. The macro average calculates the metrics independently for each class and then takes the average, while the weighted average considers the number of instances of each class in the calculation.

5. **Confusion Matrix:** The confusion matrix indicates that the model's predictions are predominantly on the diagonal cells (true positives and true negatives), indicating excellent performance. The off-diagonal cells (false positives and false negatives) contain very low values, suggesting minimal errors.

In summary, the K-Nearest Neighbors model demonstrates outstanding performance, with high accuracy and consistently high precision, recall, and F1-score for both classes. The confusion matrix further confirms the model's effectiveness in correctly classifying instances. This level of performance indicates that the KNN model is making highly accurate predictions and is well-suited for the given task.

### Logistic Regression model 

<img width="326" alt="image" src="https://github.com/Khadija-khanom/credit_card_fraud_detection2/assets/138976722/99873a27-fbcd-4722-bc27-19aa5d5fddb5">

The classification report and confusion matrix provide insights into the performance of the Logistic Regression model on the dataset:

1. **Precision, Recall, and F1-Score:** For both classes (0 and 1), the precision and recall are relatively balanced, but there is a slight difference. The model has higher precision for class 1 (0.87) compared to class 0 (0.78), and higher recall for class 0 (0.88) compared to class 1 (0.76). This indicates that the model is better at correctly identifying true negatives for class 0 and true positives for class 1. The F1-scores are around 0.83 for class 0 and 0.81 for class 1.

2. **Support:** The support indicates the number of instances of each class in the dataset. Both classes have similar support, suggesting a balanced dataset.

3. **Accuracy:** The overall accuracy of the Logistic Regression model is approximately 82%, indicating that the model correctly predicted the class of about 82% of the instances.

4. **Macro Avg and Weighted Avg:** Both macro and weighted averages for precision, recall, and F1-score are around 0.82, aligning with the performance metrics of individual classes. The macro average calculates the metrics independently for each class and then takes the average, while the weighted average considers the number of instances of each class in the calculation.

5. **Confusion Matrix:** The confusion matrix indicates that the model's predictions are spread across the diagonal and off-diagonal cells. There are a significant number of false negatives and false positives, indicating areas where the model struggles.

In summary, while the Logistic Regression model provides decent accuracy, it seems to have more difficulty with correctly identifying true positives for class 1 and true negatives for class 0. The F1-scores and confusion matrix highlight some imbalances in the model's performance. It might be worth investigating further to understand and potentially address the challenges it faces in correctly classifying instances.

### Decision Tree model 

<img width="314" alt="image" src="https://github.com/Khadija-khanom/credit_card_fraud_detection2/assets/138976722/c148c189-cac1-4fb2-83b9-96919218f010">

The classification report and confusion matrix provide insights into the performance of the Decision Tree model on the dataset:

1. **Precision, Recall, and F1-Score:** For both classes (0 and 1), precision, recall, and F1-score are all perfect with a value of 1.00. This indicates that the Decision Tree model is performing flawlessly, correctly identifying all instances of both classes.

2. **Support:** The support indicates the number of instances of each class in the dataset. Both classes have similar support, suggesting a balanced dataset.

3. **Accuracy:** The overall accuracy of the Decision Tree model is approximately 100%, indicating that the model correctly predicted the class of all instances.

4. **Macro Avg and Weighted Avg:** Both macro and weighted averages for precision, recall, and F1-score are all perfect with a value of 1.00, aligning with the flawless performance metrics of individual classes. The macro average calculates the metrics independently for each class and then takes the average, while the weighted average considers the number of instances of each class in the calculation.

5. **Confusion Matrix:** The confusion matrix indicates that the model's predictions are entirely on the diagonal cells (true positives and true negatives), demonstrating perfect performance. There are no false positives or false negatives.

In summary, the Decision Tree model demonstrates exceptional performance, with a flawless accuracy of 100% and perfect precision, recall, and F1-score for both classes. The confusion matrix further confirms that the model has correctly classified all instances. This level of performance suggests that the Decision Tree model is very well-suited for the task and is making perfect predictions on this dataset.

### Random Forest model 

<img width="349" alt="image" src="https://github.com/Khadija-khanom/credit_card_fraud_detection2/assets/138976722/bb96641d-6f46-4b5b-9ff8-b424b1f41bd3">

The classification report and confusion matrix provide insights into the performance of the Random Forest model on the dataset:

1. **Precision, Recall, and F1-Score:** For both classes (0 and 1), precision, recall, and F1-score are all perfect with a value of 1.00. This indicates that the Random Forest model is performing flawlessly, correctly identifying all instances of both classes.

2. **Support:** The support indicates the number of instances of each class in the dataset. Both classes have similar support, suggesting a balanced dataset.

3. **Accuracy:** The overall accuracy of the Random Forest model is approximately 100%, indicating that the model correctly predicted the class of all instances.

4. **Macro Avg and Weighted Avg:** Both macro and weighted averages for precision, recall, and F1-score are all perfect with a value of 1.00, aligning with the flawless performance metrics of individual classes. The macro average calculates the metrics independently for each class and then takes the average, while the weighted average considers the number of instances of each class in the calculation.

5. **Confusion Matrix:** The confusion matrix indicates that the model's predictions are almost entirely on the diagonal cells (true positives and true negatives), demonstrating perfect performance. There are only a few false positives and false negatives.

In summary, the Random Forest model demonstrates exceptional performance, with a flawless accuracy of 100% and perfect precision, recall, and F1-score for both classes. The confusion matrix further confirms that the model has correctly classified nearly all instances. This level of performance suggests that the Random Forest model is extremely well-suited for the task and is making perfect predictions on this dataset.

### Learning Curve of Machine Learning Models
**learning curve of K-Nearest Neighbors model**

![image](https://github.com/Khadija-khanom/credit_card_fraud_detection2/assets/138976722/5f7c6afa-0c9b-4951-a327-61b9591451af)


**learning curve of Logistic Regression  model**

![image](https://github.com/Khadija-khanom/credit_card_fraud_detection2/assets/138976722/c355f52b-3a3e-4e14-8bf2-fa760ae4d56d)

**learning curve of Decision Tree model**

![image](https://github.com/Khadija-khanom/credit_card_fraud_detection2/assets/138976722/b12833c6-ccae-4be3-9fa1-bba66cf48b04)

**learning curve of Random Forest model**

![image](https://github.com/Khadija-khanom/credit_card_fraud_detection2/assets/138976722/df8492e3-f3eb-4d66-af99-2f398c417d76)


## Implementation Of Deep learning model 

### Learning Algorithm for CNN (Convolutional Neural Network)

**Load and preprocess the dataset:**

- Load dataset 'processed.csv'.

- Handle missing values.

- Standardize features.

**Balance class distribution using SMOTE:** Apply SMOTE to the dataset to address class imbalance.

**Shuffle the data:** Randomly shuffle the data for randomness.

**Reshape data for CNN:** Reshape feature data for training and testing (X_train, X_test).

**Split data into training and testing sets:** Split data into X_train, y_train (80%) and X_test, y_test (20%).

**Create a Sequential CNN model:**

Create model_cnn.

- Add Conv1D layer with 64 filters, kernel size 3, ReLU activation, and input shape.

- Add MaxPooling1D layer with pool size 2.

- Add Flatten layer.

- Add Dense layer with 128 neurons and ReLU activation.

- Add Dropout layer with dropout rate 0.5.

- Add Output Dense layer with 1 neuron and sigmoid activation.

**Compile the CNN model:** Compile model_cnn with binary cross-entropy loss and Adam optimizer.

**Train the CNN model:** For each epoch, perform a forward pass, calculate loss, perform backward pass (backpropagation), and update weights and biases using Adam optimizer.

**Model Evaluation:**
- For each data point in X_test, predict using model_cnn.

- Convert predicted probabilities to binary predictions (threshold = 0.5).

**Calculate total accuracy and testing accuracy:** Calculate accuracy_score(y_test, y_pred_cnn).

**Generate evaluation metrics:**

- Generate a classification report (precision, recall, F1-score).

- Generate a confusion matrix (true positives, true negatives, false positives, false negatives).

**Plot Learning Curve:** Plot a learning curve to visualize training and testing accuracy over epochs.

### Learning Algorithm for RNN (Recurrent Neural Network)

**Load and preprocess the dataset:**

- Load dataset 'processed.csv'.
- Handle missing values.
- Standardize features.

**Balance class distribution using SMOTE:** Apply SMOTE to the dataset to address class imbalance.

**Shuffle the data:** Randomly shuffle the data for randomness.

**Reshape data for RNN:** Reshape feature data for training and testing (X_train_rnn, X_test_rnn).

**Split data into training and testing sets:**
Split data into X_train_rnn, y_train_rnn (80%) and X_test_rnn, y_test_rnn (20%).

**Create a Sequential RNN model:**

Create model_rnn.

- Add LSTM layer with 64 units and ReLU activation, specifying input shape.

- Add Dense layer with 128 neurons and ReLU activation.

- Add Dropout layer with dropout rate 0.5.

- Add Output Dense layer with 1 neuron and sigmoid activation.

**Compile the RNN model:** Compile model_rnn with binary cross-entropy loss and Adam optimizer.

**Train the RNN model:**
For each epoch, perform a forward pass, calculate loss, perform backward pass (backpropagation), and update weights and biases using Adam optimizer.

**Model Evaluation:**

- For each data point in X_test_rnn, predict using model_rnn.
- Convert predicted probabilities to binary predictions (threshold = 0.5).

**Calculate total accuracy and testing accuracy:**

Calculate accuracy_score(y_test_rnn, y_pred_rnn).

**Generate evaluation metrics:**

- Generate a classification report (precision, recall, F1-score).
- Generate a confusion matrix (true positives, true negatives, false positives, false negatives).

**Plot Learning Curve:**
Plot a learning curve to visualize training and testing accuracy over epochs.


## Evaluating the performance of deep learning models

### CNN (Convolutional Neural Network)

<img width="296" alt="image" src="https://github.com/Khadija-khanom/credit_card_fraud_detection2/assets/138976722/cfe2d9c5-3a62-40e1-b615-2d99e2735a2f">


The classification report and confusion matrix provide insights into how well CNN model performed on the dataset:

1. **Precision, Recall, and F1-Score:** For both classes (0 and 1), precision, recall, and F1-score are all around 0.98. This indicates that the model is performing consistently well for both classes, achieving a balance between identifying true positives (recall) and minimizing false positives (precision).

2. **Support:** The support indicates the number of instances of each class in the dataset. Both classes have similar support, suggesting a balanced dataset.

3. **Accuracy:** The overall accuracy of the model is approximately 98%, indicating that the model correctly predicted the class of about 98% of the instances.

4. **Macro Avg and Weighted Avg:** Both macro and weighted averages for precision, recall, and F1-score are around 0.98, which align with the performance metrics of individual classes. The macro average calculates the metrics independently for each class and then takes the average, while the weighted average considers the number of instances of each class in the calculation.

5. **Confusion Matrix:** The confusion matrix shows the distribution of predictions across the four possible outcomes: true positives, true negatives, false positives, and false negatives. The model has higher counts in the diagonal cells (true positives and true negatives), indicating that it is performing well overall. The off-diagonal cells (false positives and false negatives) are relatively small, suggesting that the model's errors are limited.

In summary, the CNN model demonstrates high accuracy and consistent performance across both classes, as evidenced by the precision, recall, and F1-score. The confusion matrix confirms the model's ability to correctly classify a majority of instances while making only a small number of errors.

###  RNN (Recurrent Neural Network)

<img width="321" alt="image" src="https://github.com/Khadija-khanom/credit_card_fraud_detection2/assets/138976722/7609c535-6918-482b-a559-8ba247b904ae">


The classification report and confusion matrix provide insights into the performance of RNN model on the dataset:

1. **Precision, Recall, and F1-Score:** For both classes (0 and 1), precision, recall, and F1-score are all around 0.99. This indicates that the model is performing extremely well for both classes, with a high ability to identify true positives (recall) and minimize false positives (precision).

2. **Support:** The support indicates the number of instances of each class in the dataset. Both classes have similar support, suggesting a balanced dataset.

3. **Accuracy:** The overall accuracy of the model is approximately 99%, indicating that the model correctly predicted the class of about 99% of the instances.

4. **Macro Avg and Weighted Avg:** Both macro and weighted averages for precision, recall, and F1-score are around 0.99, aligning with the performance metrics of individual classes. The macro average calculates the metrics independently for each class and then takes the average, while the weighted average considers the number of instances of each class in the calculation.

5. **Confusion Matrix:** The confusion matrix indicates that the model's predictions are predominantly on the diagonal cells (true positives and true negatives), indicating excellent performance. The off-diagonal cells (false positives and false negatives) contain relatively small values, suggesting minimal errors.

In summary, the RNN model demonstrates exceptional performance, with high accuracy and consistently high precision, recall, and F1-score for both classes. The confusion matrix further validates the model's effectiveness in correctly classifying instances. This level of performance indicates that the RNN model is making highly accurate predictions and is suitable for the task at hand.

### Learning Curve of Deep Learning Model

**CNN (Convolutional Neural Network) Model:**

![image](https://github.com/Khadija-khanom/credit_card_fraud_detection2/assets/138976722/a8e87563-8836-4548-9e5c-0918503817fd)



**RNN (Recurrent Neural Network) Model:**
![image](https://github.com/Khadija-khanom/credit_card_fraud_detection2/assets/138976722/c7418528-d730-4759-9121-cd244db1e459)


## comparative analysis Between Machine learning models and Deep learning models

Here's an overall table that includes the performance metrics of all the models: CNN, RNN, K-Nearest Neighbors (KNN), Logistic Regression, Decision Tree, and Random Forest.

**Comparative Analysis:**

Based on the provided performance metrics, we can make the following comparative analysis among the models:

1. **Accuracy:** Both the Decision Tree and Random Forest models achieved the highest accuracy of 1.00, indicating that they were able to correctly classify all instances in the dataset. The RNN model also had a high accuracy of 0.99, followed by K-Nearest Neighbors with an accuracy of 0.992.

2. **Precision and Recall:** The Decision Tree and Random Forest models achieved perfect precision and recall for both classes, indicating that they correctly identified all instances of both classes. The RNN model also demonstrated high precision and recall, with slight differences in recall between classes. K-Nearest Neighbors had slightly imbalanced precision and recall for class 0 and class 1.

3. **F1-Score:** The F1-scores for the Decision Tree and Random Forest models were perfect, reflecting their flawless performance. The RNN model had high F1-scores, and the K-Nearest Neighbors model had high F1-scores, but they were slightly lower compared to the Decision Tree and Random Forest.

**Conclusion:**

<img width="668" alt="image" src="https://github.com/Khadija-khanom/credit_card_fraud_detection2/assets/138976722/f9692443-b8d9-4a4a-8a12-150f517eb809">


In terms of performance, the Decision Tree and Random Forest models stand out as the top performers, achieving perfect accuracy, precision, recall, and F1-scores. The RNN model also performed exceptionally well with high accuracy and balanced precision and recall. The K-Nearest Neighbors model achieved high accuracy but had slightly imbalanced precision and recall for class 0 and class 1. The CNN model had a very good performance but showed a slightly lower accuracy compared to the others.

The choice of the best model depends on the specific goals of task. If we prioritize accuracy and balanced performance, the Decision Tree or Random Forest models might be suitable. If we require sequential data processing, the RNN model would be a strong choice. It's important to consider the nature of data, interpretability, computational resources, and other factors when selecting the best model for your application.
