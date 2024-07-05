# Summer_Internship
## WEEK-1</br>
### Day 1: Filling Missing Values </br>
On the first day, I focused on handling missing values in the dataset. This is a crucial step in data preprocessing as missing values can significantly impact the performance of machine learning models. I learned various techniques to handle missing data, such as: </br>
•	Imputation with mean/median/mode: Replacing missing values with the mean, median, or mode of the column.</br>
•	Forward/Backward Fill: Using the previous or next value to fill the missing entries.</br>
•	Using a model to predict missing values: Building a simple model to predict and fill missing data based on other features.</br>
### Day 2: Encoding Columns </br>
Encoding categorical variables is essential for machine learning models. I explored different encoding techniques: </br>
2.1 One-Hot Encoding </br>
This method converts categorical variables into a series of binary columns. Each category is represented as a column, and a 1 or 0 is placed in the column to indicate the presence of the category.</br>
2.2 Label Encoding </br>
Label encoding converts categorical values into integer labels. Each unique category is assigned a unique integer.</br>
2.3 Ordinal Encoding</br>
Ordinal encoding is used when the categorical data has an intrinsic order. Categories are encoded as ordered integers.</br>
2.4 Target Encoding</br>
Target encoding replaces a categorical value with the mean of the target variable for that category. This is useful when there is a strong relationship between the categorical variable and the target.</br>
### Day 3: Splitting Dataset into Test/Train Data</br>
Splitting the dataset is vital to evaluate the model's performance. I practiced:</br>
3.1 Cross-Validation</br>
Cross-validation involves splitting the dataset into multiple folds and training the model multiple times. This helps in obtaining a more reliable estimate of model performance. I implemented k-fold cross-validation to ensure my model's robustness.</br>
### Day 4: Feature Scaling</br>
Feature scaling ensures that all features contribute equally to the model. I experimented with different scaling techniques:</br>
4.1 Standardization</br>
Standardization involves rescaling the features to have a mean of 0 and a standard deviation of 1. This is essential for algorithms that assume normally distributed data.</br>
4.2 Min-Max Scaling</br>
Min-max scaling transforms features to a fixed range, typically [0, 1]. This scaling is useful when the data has varying ranges.</br>
4.3 Robust Scaling</br>
Robust scaling uses the median and interquartile range, making it robust to outliers.</br>
4.4 Normalization</br>
Normalization scales the data to have a unit norm (magnitude of 1). This technique is helpful for algorithms that compute distances between data points.</br>
### Day 5: Time Series Analysis</br>
I delved into time series analysis and forecasting using ARIMA and SARIMA models.</br>
5.1 ARIMA (AutoRegressive Integrated Moving Average)</br>
ARIMA models are used for analyzing and forecasting time series data. The ARIMA model combines three components:</br>
•	AR (AutoRegression): The model uses the dependent relationship between an observation and some number of lagged observations.</br>
•	I (Integrated): This involves differencing the observations to make the time series stationary.</br>
•	MA (Moving Average): The model uses the dependency between an observation and a residual error from a moving average model applied to lagged observations.</br>
5.2 SARIMA (Seasonal ARIMA)</br>
SARIMA extends ARIMA by adding seasonal components. It captures seasonality in the data through additional seasonal terms. This model is useful for datasets with strong seasonal patterns.</br>
                                    
## WEEK 2</br>
### Day 1: Model Identification</br>
We started the week by working on model identification for an assigned dataset. The tasks included:</br>
1.1 Run Sequence</br>
A run sequence plot helps visualize the data over time, providing insights into trends, shifts, or patterns that may affect the model. I learned how to create and interpret run sequence plots to identify underlying data structures.</br>
1.2 Auto Correlation Plot</br>
Auto Correlation Plots (ACF) are essential in identifying the correlation between observations of a time series at different lags. I generated ACF plots to detect any patterns or correlations in the dataset that would inform the model selection.</br>
### Day 2: Model Estimation</br>
Model estimation involves determining the parameters of the chosen model that best fit the data. I focused on:</br>
•	Understanding different estimation techniques.</br>
•	Using software tools to estimate model parameters.</br>
•	Interpreting the estimated parameters to ensure they make sense and improve the model.</br>
### Day 3: Model Validation</br>
Model validation is crucial to ensure that the model generalizes well to new data. I engaged in:</br>
•	Splitting data into training and validation sets.</br>
•	Applying cross-validation techniques to validate the model's performance.</br>
•	Analyzing validation metrics to assess the model's accuracy and reliability.</br>
### Day 4: Predicting Mortality</br>
I worked on a given dataset to predict mortality rates. The tasks included:</br>
2.1 Data Cleaning</br>
Data cleaning is a critical step to ensure the dataset is free from inconsistencies, missing values, and errors. I performed:</br>
•	Removing or imputing missing values.</br>
•	Correcting data entry errors.</br>
•	Normalizing data formats for consistency.</br>
2.2 Data Analysis Using Plots</br>
Visual data analysis helps in understanding data distributions and relationships. I created various plots such as:</br>
•	Histograms to view data distributions.</br>
•	Scatter plots to identify relationships between variables.</br>
•	Box plots to detect outliers.</br>
### Day 5: Logistic Regression Classifier</br>
We focused on implementing a logistic regression model to predict mortality:</br>
2.3 Using Logistic Regression Classifier</br>
Logistic regression is a powerful classification algorithm used for binary outcomes. I:</br>
•	Prepared the dataset for logistic regression.</br>
•	Trained the logistic regression model on the cleaned data.</br>
•	Evaluated the model's performance using accuracy, precision, recall, and F1 score.</br>
2.4 ROC Curve</br>
The ROC curve is a graphical representation of a classifier's performance. I:</br>
•	Plotted the ROC curve to visualize the trade-off between sensitivity (true positive rate) and specificity (false positive rate).</br>
•	Calculated the Area Under the Curve (AUC) to quantify the model's performance. A higher AUC indicates better model performance.</br>

## WEEK-3</br>
Project: ICU Mortality Rate Prediction</br></br>
### Day 1: Reading in the Data</br>
The week began with loading the ICU dataset into the working environment. This step involved:</br>
•	Understanding the dataset structure and features.</br>
•	Using Python libraries such as Pandas to read and manipulate the data.</br>
•	Checking for any missing values or inconsistencies in the dataset to prepare it for further analysis.</br>
### Day 2: Analyzing the Data by Plotting Required Plots</br>
Data visualization is critical for gaining insights into the dataset. I focused on creating various plots to analyze the data:</br>
•	Histograms: To understand the distribution of numerical variables.</br>
•	Scatter Plots: To identify relationships between different features.</br>
•	Box Plots: To detect outliers and understand the spread of the data.</br>
•	Correlation Matrix: To see the correlation between different variables and identify any multicollinearity issues.</br>
### Day 3: Running Logistic Regression</br>
Logistic regression was chosen to predict ICU mortality rates. The steps included:</br>
•	Preparing the Data: Splitting the dataset into training and test sets, and applying necessary transformations such as encoding categorical variables.</br>
•	Training the Model: Implementing logistic regression using libraries like scikit-learn.</br>
•	Evaluating the Model: Using metrics such as accuracy, precision, recall, and F1 score to evaluate the model's performance on the training data.</br>
### Day 4: Creating an ROC Curve</br>
To further evaluate the model, I created an ROC curve:</br>
•	Plotting the ROC Curve: Visualizing the trade-off between the true positive rate (sensitivity) and false positive rate (1-specificity) across different threshold values.</br>
•	Calculating the AUC: The Area Under the ROC Curve (AUC) was calculated to quantify the overall performance of the model. A higher AUC indicates a better-performing model.</br>
### Day 5: Finalizing the Project</br>
The final day was dedicated to compiling the findings and refining the model:</br>
•	Model Tuning: Making any necessary adjustments to improve the model based on the ROC curve and evaluation metrics.</br>
•	Documentation: Documenting the entire process, including the methods used, results obtained, and insights gained from the analysis.</br>
•	Presentation: Preparing a presentation to showcase the project results and discussing potential improvements or next steps.</br>

## WEEK 4</br>
Project: American Sign Language (ASL) Recognition</br>
### Day 1: Project Introduction and Data Collection</br>
The final week began with an introduction to the ASL recognition project. Key activities included:</br>
•	Understanding Project Scope: Discussing the objectives and goals of the ASL recognition project.</br>
•	Data Collection: Gathering and preprocessing a dataset containing images or videos of ASL gestures. This involved using pre-existing datasets or collecting new data if needed.</br>
### Day 2: Data Preprocessing</br>
Data preprocessing is crucial for image recognition tasks. The steps included:</br>
•	Image Resizing: Ensuring all images are of a uniform size for model consistency.</br>
•	Normalization: Scaling pixel values to a range of [0, 1] to facilitate faster convergence during model training.</br>
•	Augmentation: Applying techniques like rotation, flipping, and zooming to increase the diversity of the training data and improve model robustness.</br>
### Day 3: Model Building</br>
Building the ASL recognition model involved selecting and implementing an appropriate machine learning or deep learning architecture:</br>
•	Choosing the Model: Deciding on a Convolutional Neural Network (CNN) for image recognition due to its effectiveness in handling visual data.</br>
•	Model Implementation: Using frameworks like TensorFlow or PyTorch to build the CNN.</br>
•	Training the Model: Feeding the preprocessed data into the model and training it using backpropagation and gradient descent.</br>
### Day 4: Model Evaluation and Tuning</br>
Evaluating and fine-tuning the model to improve its accuracy and performance:</br>
•	Evaluation Metrics: Using metrics like accuracy, precision, recall, and F1 score to evaluate the model's performance on a validation set.</br>
•	Hyperparameter Tuning: Adjusting parameters like learning rate, batch size, and number of epochs to optimize model performance.</br>
•	Cross-Validation: Applying cross-validation techniques to ensure the model generalizes well to unseen data.</br>
### Day 5: Finalizing and Presenting the Project</br>
The last day was dedicated to finalizing the ASL recognition project and preparing for the presentation:</br>
•	Model Deployment: Discussing potential deployment strategies for the ASL recognition system.</br>
•	Documentation: Compiling a detailed report on the project, including data preprocessing steps, model architecture, training process, and evaluation results.</br>
•	Presentation: Preparing a presentation to showcase the project findings, challenges faced, and potential future improvements.</br>
