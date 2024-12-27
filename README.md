# AF7. Scheduling with Apache Airflow/Scheduled Model Retraining and Batch Prediction

Created: December 10, 2024 8:47 PM

# Objective:

In this lab, we are going to retrain a machine learning model over the decision based on batch prediction and show the scheduling process using Apache Airflow exposing through Poridhi Load Balancer.

# Table of contents:

1. Basic overviews on Machine Learning Model Retraining.
2. Building a model and create the whole retraining process.
3. Scheduling it with DAG creation of Apache Airflow.
4. Containerise using Docker and expose it with Load Balancer.

### **What is Machine Learning Model Retraining?**

A ML model is basically created and trained using some specific dataset offline. After doing all the lifecycle of Data Science, finally it gives some better accuracy and then the model is deployed. But the usual scenario of the machine learning projects is that it receives the live data continuously. When the new datas are coming the model is in need of training with respect to the new datas continuously.

![model_retraining-2.svg](AF7%20Scheduling%20with%20Apache%20Airflow%20Scheduled%20Model%20210fd9ab443a46c9aa85872ff23945c6/model_retraining-2.svg)

### Step 01 : Open VS Code and create the files that needed & Create a virtual environment.

- Update and upgrade the sudo package list.
    
    ```markup
    sudo apt update
    ```
    
    ```markup
    sudo apt upgrade -y
    ```
    
- Install python3-venv.
    
    ```markup
    sudo apt install python3-venv
    ```
    
- Create virtual environment.
    
    ```markup
    python3 -m venv retrain
    ```
    
    ```markup
    source retrain/bin/activate
    ```
    
- Install Apache Airflow.

```markup
AIRFLOW_VERSION=2.7.3
PYTHON_VERSION="$(python3 --version | cut -d " " -f 2 | cut -d "." -f 1-2)"
CONSTRAINT_URL="[https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt](https://raw.githubusercontent.com/apache/airflow/constraints-$%7BAIRFLOW_VERSION%7D/constraints-$%7BPYTHON_VERSION%7D.txt)"
pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"
```

### Step 02 : Initilaise the `docker-compose.yaml` file and `Dockerfile`.

- Use this Cli to create the docker compose yaml file.

```python
curl -LfO '[https://airflow.apache.org/docs/apache-airflow/2.10.3/docker-compose.yaml](https://airflow.apache.org/docs/apache-airflow/2.10.3/docker-compose.yaml)'
```

- Will create a yaml file automatically to your directory and get this output.
    
    ![image-01.png](AF7%20Scheduling%20with%20Apache%20Airflow%20Scheduled%20Model%20210fd9ab443a46c9aa85872ff23945c6/image-01.png)
    

or, can use this as docker-compose file.

```
# docker-compose.yml
version: '3'
services:
  airflow:
    build: .
    restart: always
    environment:
      - AIRFLOW__CORE__EXECUTOR=LocalExecutor
      - AIRFLOW__CORE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres:5432/airflow
      - AIRFLOW__CORE__LOAD_EXAMPLES=False
    user: "${AIRFLOW_UID:-50000}:0"
    volumes:
      - ./dags:/opt/airflow/dags
      - ./logs:/opt/airflow/logs:rw
    ports:
      - "8081:8080"
    depends_on:
      - postgres
    command: bash -c "airflow db init && airflow users create --username admin --password admin --firstname Anonymous --lastname Admin --role Admin --email admin@example.com && airflow webserver & airflow scheduler"

  postgres:
    image: postgres:13
    environment:
      - POSTGRES_USER=airflow
      - POSTGRES_PASSWORD=airflow
      - POSTGRES_DB=airflow
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

- Make some directories in it.

```markup
mkdir -p ./dags ./logs ./plugins ./config
```

- `Dockerfile`
    
    ![image-02.png](AF7%20Scheduling%20with%20Apache%20Airflow%20Scheduled%20Model%20210fd9ab443a46c9aa85872ff23945c6/image-02.png)
    
- `requirements.txt file`
    
    ```markup
    apache-airflow==2.7.3
    ```
    

### Step 03 : Write the code.

Let’s try to build a model which will generate synthetic data to train a model and test it to save the accuracy, where we have saved both the data and the model in separate directories. Then we made a batch prediction of the model with new data and checked the threshold or accuracy. The model will be retrained depending on the decision made based on the accuracy condition given to the model.

```python
import os

# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
#from airflow.operators.dummy_operators import DummpyOperator
from datetime import datetime , timedelta

# Creation of two directories to store the data and model
DATA_DIR= '/opt/airflow/data'
MODEL_DIR= 'opt/airflow/model'

# Defining the functions of the whole process
def data_preparation(**context):
    #Generate synthetic data for demonstration
    np.random.seed(42)
    X= np.random.rand(1000,5)  #5 features
    y= (X[:,0] + X[:,1] > 1).astype(int) # Binary classification

    # Split the data
    X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=42)

    # scale the feature
    scaler= StandardScaler()
    X_train_scaled= scaler.fit_transform(X_train)
    X_test_scaled= scaler.transform(X_test)

    #save data to files
    os.makedirs(DATA_DIR, exist_ok=True)
    np.save(os.path.join(DATA_DIR, 'X_train.npy'), X_train_scaled)
    np.save(os.path.join(DATA_DIR, 'X_test.npy'), X_test_scaled)
    np.save(os.path.join(DATA_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(DATA_DIR, 'y_test.npy'), y_test)
    
    #save scaler for future use
    from joblib import dump
    dump(scaler, os.path.join(DATA_DIR, 'current_scaler.joblib'))

    context['task_instance'].xcom_push(key='data_prepared', value=True)

def model_training(**context):

    #create model directory
    os.makedirs(MODEL_DIR, exist_ok=True)

    #Load prepared data
    X_train= np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    y_train= np.load(os.path.join(DATA_DIR, 'y_train.npy'))

    # Training Random Forest classifier
    model= RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    #save the model
    from joblib import dump
    dump(model, os.path.join(MODEL_DIR, 'current_model.joblib'))

    context['task_instance'].xcom_push(key='model_trained', value=True)

    

def model_evaluation(**context):

    #Load data
    X_test= np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    y_test= np.load(os.path.join(DATA_DIR, 'y_test.npy'))

    #Load model
    from joblib import load
    model= load(os.path.join(MODEL_DIR, 'current_model.joblib'))

    #Make prediction
    y_pred= model.predict(X_test)

    #calculate metrics
    accuracy= accuracy_score(y_test, y_pred)
    f1= f1_score(y_test, y_pred, average='weighted')
    report= classification_report(y_test, y_pred)

    #Log metrics
    print(f"Model Accuracy : {accuracy}")
    print(f"f1 score : {f1_score}")
    print(f"Classification Report : {classification_report}")

    # Push metrics to XCom for decision making
    context['task_instance'].xcom_push(key='model_accuracy', value=accuracy)
    context['task_instance'].xcom_push(key='model_f1_score', value=f1)

def batch_prediction(**context):

    # Load the scaler
    from joblib import load
    scaler= load(os.path.join(MODEL_DIR, 'current_scaler.joblib'))

    np.random.seed(int(datetime.now().timestamp))
    new_data= np.random.rand(100,5)

    # Load the current model and predict
    model= load(os.path.join(MODEL_DIR, 'current_model.joblib'))

    #scale the new data
    new_data_scaled=scaler.transform(new_data)

    new_labels= (new_data[:,0] + new_data[:,1] > 1).astype(int)

    # predict on new data
    predictions= model.predict(new_data_scaled)

    #calculate the batch prediction metrics
    batch_accuray = accuracy_score(new_labels, predictions)
    batch_f1= f1_score(new_labels, predictions, average='weighted')

    print(f"Batch accuracy: {batch_accuray}")
    print(f"batch f1 score: {batch_f1}")

    # Save prediction and metrics
    np.save(os.path.join(DATA_DIR, 'batch_prediction.npy'), predictions)
    np.save(os.path.join(DATA_DIR, 'batch_labels.npy'), new_labels)

    # push batch prediction metrics to xcom
    context['task_instance'].xcom_push(key='batch_accuracy', value=batch_accuray)
    context['task_instance'].xcom_push(key='batch_f1_score', value=batch_f1)

def decide_retraining(**context):

    '''
    Decide whether to retrain the model based on batch prediction performance
    retrain conditions:
    1. batch accuracy drops more than 10% than the original model accuracy
    2. batch f1 score drops by more than 15%
    '''

    # Retrieve metrics from xcom
    model_accuracy= context['task_instance'].xcom_pull(task_ids='evaluate_model', key='model_accuracy')
    model_f1_score= context['task_instance'].xcom_pull(task_id='evaluate_model', key='model_f1_score')

    batch_accuracy= context['task_instance'].xcom_pull(task_ids='batch_prediction', key='batch_accuracy')
    batch_f1_score= context['task_instance'].xcom_pull(task_id='batch_prediction', key='batch_f1_score')

    # Define retraining conditions
    accuracy_threshold= model_accuracy * 0.9
    f1_threshold = model_f1_score * 0.85

    if batch_accuracy < accuracy_threshold or batch_f1_score < f1_threshold:
        print("Model performance degraded. Initiating retraining")
        return 'retrain_model'
    else:
        print("Model performance satisfactory. No retraining needed")
        return 'no_retraining'

def retrain_model(**context):
    '''
    Retrain the model with more datas
    '''

    # Load existing training data
    X_train= np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    y_train= np.load(os.path.join(DATA_DIR, 'y_train.npy'))

    # Load batch prediction data
    batch_prediction_data= np.load(os.path.join(DATA_DIR, 'batch_prediction.npy'))
    batch_labels= np.load(os.path.join(DATA_DIR, 'batch_labels.npy'))

    #Combine original and batch data
    X_retrain= np.vstack(X_train, batch_prediction_data)
    y_retrain= np.concatenate([y_train, batch_labels])

    # Retrain with slightly high parameters
    model= RandomForestClassifier(
        n_estimators=120,
        max_depth=None,
        min_samples_split=5,
        random_state=42
    )
    model.fit(X_retrain,y_retrain)

    #save the retrain model
    from joblib import dump
    dump(model, os.path.join(MODEL_DIR, 'retrained_model.joblib'))

def no_retrain_model():

    pass

# DAG creation and scheduling the tasks

with DAG(
    'ml_model_retraining',
    default_args={
        'owner': 'nakib',
        'start_date': datetime(2024,1,1),
        'retries': 3,
        'retry_delay': timedelta(minutes=5)
    },
    schedule_interval='@weekly'
) as dag: 

    prepare_data=PythonOperator(
        task_id='data_preparation',
        python_callable=data_preparation
    )

    train_model=PythonOperator(
        task_id= 'model_training',
        python_callable=model_training
    )

    evaluate_model=PythonOperator(
        task_id= 'model_evaluation',
        python_callable=model_evaluation
    )

    batch_predict=PythonOperator(
        task_id='batch_prediction',
        python_callable=batch_prediction
    )

    retrain_decision=PythonOperator(
        task_id='decide_retraining',
        python_callable=decide_retraining
    )

    retrain_model_task=PythonOperator(
        task_id='retrain_model',
        python_callable=retrain_model
    )

    no_retraining_task= BranchPythonOperator(
        task_id='no_retrain_model',
        python_callable=no_retrain_model
    )

    

    prepare_data >> train_model >> evaluate_model >> batch_predict >> retrain_decision 
    retrain_decision >> [retrain_model_task, no_retraining_task]

```

### Step 04: Build the file with docker-compose.

```markup
docker-compose up —build
```

![image-03.png](AF7%20Scheduling%20with%20Apache%20Airflow%20Scheduled%20Model%20210fd9ab443a46c9aa85872ff23945c6/image-03.png)

```markup
docker ps
```

![image-04.png](AF7%20Scheduling%20with%20Apache%20Airflow%20Scheduled%20Model%20210fd9ab443a46c9aa85872ff23945c6/image-04.png)

### Step 05: Expose the Airflow UI using Load Balancer.

- Find the local ip using the command.
    
    ```markup
    ip addr show eth0
    ```
    
    ![image-05.png](AF7%20Scheduling%20with%20Apache%20Airflow%20Scheduled%20Model%20210fd9ab443a46c9aa85872ff23945c6/image-05.png)
    

or by using terminal

![image-06.png](AF7%20Scheduling%20with%20Apache%20Airflow%20Scheduled%20Model%20210fd9ab443a46c9aa85872ff23945c6/image-06.png)

- Go to the load balancer.
    
    ![image-07.png](AF7%20Scheduling%20with%20Apache%20Airflow%20Scheduled%20Model%20210fd9ab443a46c9aa85872ff23945c6/image-07.png)
    

- Create the Load balancer.
    
    ![image-08.png](AF7%20Scheduling%20with%20Apache%20Airflow%20Scheduled%20Model%20210fd9ab443a46c9aa85872ff23945c6/image-08.png)
    

Load Balancer created

![image-10.png](AF7%20Scheduling%20with%20Apache%20Airflow%20Scheduled%20Model%20210fd9ab443a46c9aa85872ff23945c6/image-10.png)

And launch it.

**Airflow UI view:**

DAG created.

![image-11.png](AF7%20Scheduling%20with%20Apache%20Airflow%20Scheduled%20Model%20210fd9ab443a46c9aa85872ff23945c6/image-11.png)

DAG triggered successfully.

![image-12.png](AF7%20Scheduling%20with%20Apache%20Airflow%20Scheduled%20Model%20210fd9ab443a46c9aa85872ff23945c6/image-12.png)

Scheduled tasks.

![image-13.png](AF7%20Scheduling%20with%20Apache%20Airflow%20Scheduled%20Model%20210fd9ab443a46c9aa85872ff23945c6/image-13.png)

# Conclusion:

In this lab, we successfully created the Scheduling process of Machine Learning Model Retraining with Apache Airflow and launched it using **Poridhi Load Balancer**.