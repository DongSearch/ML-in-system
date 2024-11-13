In this assignment they have to train a machine learning model on a real dataset in the Raspberry PI and evaluate in test

1. Download Anaconda ([What is Anaconda?](https://docs.anaconda.com/distro-or-miniconda/))on your computer - [link to ARM version with Python 3.12](https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-aarch64.sh) 
2. Copy the anaconda installer to the Jetson nano/Raspberry PI via scp

gidong@DESKTOP-UBH17S5:~/ex1$ scp Anaconda3-2024.10-1-Linux-aarch64.sh pi@169.254.68.27:~
pi@169.254.68.27's password:
Anaconda3-2024.10-1-Linux-aarch64.sh                                                    0% 5424KB 

gidong@DESKTOP-UBH17S5:~/ex1$ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no pi@169.254.68.27
pi@169.254.68.27's password:
Linux raspberrypi 6.6.51+rpt-rpi-v8 #1 SMP PREEMPT Debian 1:6.6.51-1+rpt3 (2024-10-08) aarch64


3. Install Anaconda. It is highly recommended to install via terminal by just executing the downloaded file. 
	- During installation you can decide if you want to have anaconda at startup - I would recommend to don't do that to avoid system breaks

pi@raspberrypi:~ $ chmod +x Anaconda3-2024.10-1-Linux-aarch64.sh
pi@raspberrypi:~ $ ./Anaconda3-2024.10-1-Linux-aarch64.sh
pi@raspberrypi:~ $ source anaconda3/bin/activate

 4. Download the wine dataset from the url `https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data` with `wget` command. The dataset is in csv Format

wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data

 5. Copy the dataset to the Jetson nano/Raspberry PI via scp
gidong@DESKTOP-UBH17S5:~/ex1$ scp wine.data pi@169.254.68.27:~

6. Train a RandomForestClassifier with the following script:```
7. Edit the script such that:
	- It saves into a file the output of the function `classification_report` with name `classfication_report_{model_name}.txt` where `model_name` is the name of the model (e.g. random_forest)
	- It trains Logistic regression, KNN, 2-Layers neural network with 50 neurons for each hidden layer


import numpy as np
>>> from sklearn.model_selection import train_test_split
>>> from sklearn.preprocessing import StandardScaler
>>> from sklearn.ensemble import RandomForestClassifier
>>> from sklearn.linear_model import LogisticRegression
>>> from sklearn.neighbors import KNeighborsClassifier
>>> from sklearn.neural_network import MLPClassifier
>>> from sklearn.metrics import classification_report
>>> import joblib
>>>
>>> # Define column names and load dataset
>>> column_names = ['class', 'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash',
...                 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
...                 'proanthocyanins', 'color_intensity', 'hue',
...                 'od280/od315_of_diluted_wines', 'proline']
>>> data = pd.read_csv('wine.data', names=column_names)
>>>
>>> # Split data into features and target
>>> X = data.drop('class', axis=1)
>>> y = data['class']
>>>
>>> # Split data into train and test sets
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
>>>
>>> # Scale the features
>>> scaler = StandardScaler()
>>> X_train_scaled = scaler.fit_transform(X_train)
>>> X_test_scaled = scaler.transform(X_test)
>>>
>>> # Define function to train model, print and save classification report
>>> def train_and_evaluate(model, model_name):
...     model.fit(X_train_scaled, y_train)
...     y_pred = model.predict(X_test_scaled)
...
...     # Generate and save the classification report
...     report = classification_report(y_test, y_pred)
...     print(f"\nClassification Report for {model_name}:\n", report)
...
...     filename = f"classification_report_{model_name}.txt"
...     with open(filename, "w") as file:
...         file.write(report)
...     print(f"Classification report saved as {filename}")
...
...     # Save the trained model
...     joblib.dump(model, f'{model_name}_model.joblib')
...
>>> # Train and evaluate Random Forest
>>> rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
>>> train_and_evaluate(rf_model, "random_forest")

Classification Report for random_forest:
               precision    recall  f1-score   support

           1       1.00      1.00      1.00        14
           2       1.00      1.00      1.00        14
           3       1.00      1.00      1.00         8

    accuracy                           1.00        36
   macro avg       1.00      1.00      1.00        36
weighted avg       1.00      1.00      1.00        36

Classification report saved as classification_report_random_forest.txt
>>>
>>> # Train and evaluate Logistic Regression
>>> lr_model = LogisticRegression(random_state=42, max_iter=1000)
>>> train_and_evaluate(lr_model, "logistic_regression")

Classification Report for logistic_regression:
               precision    recall  f1-score   support

           1       1.00      1.00      1.00        14
           2       1.00      1.00      1.00        14
           3       1.00      1.00      1.00         8

    accuracy                           1.00        36
   macro avg       1.00      1.00      1.00        36
weighted avg       1.00      1.00      1.00        36

Classification report saved as classification_report_logistic_regression.txt
>>>
>>> # Train and evaluate K-Nearest Neighbors
>>> knn_model = KNeighborsClassifier(n_neighbors=5)
>>> train_and_evaluate(knn_model, "knn")

Classification Report for knn:
               precision    recall  f1-score   support

           1       0.93      1.00      0.97        14
           2       1.00      0.86      0.92        14
           3       0.89      1.00      0.94         8

    accuracy                           0.94        36
   macro avg       0.94      0.95      0.94        36
weighted avg       0.95      0.94      0.94        36

Classification report saved as classification_report_knn.txt
>>>
>>> # Train and evaluate 2-Layer Neural Network
>>> mlp_model = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
>>> train_and_evaluate(mlp_model, "neural_network")

Classification Report for neural_network:
               precision    recall  f1-score   support

           1       1.00      1.00      1.00        14
           2       1.00      1.00      1.00        14
           3       1.00      1.00      1.00         8

    accuracy                           1.00        36
   macro avg       1.00      1.00      1.00        36
weighted avg       1.00      1.00      1.00        36