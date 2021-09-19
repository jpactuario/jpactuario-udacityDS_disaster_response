# Disaster Response Pipeline Project

## Summary

The goal of this project is to build a web application to classify a message into categories that aid workers can work on e.g.
whether the sender is requesting for assistance and, if so, on which domain e.g. water, shelter, food, clothing, etc.

The labelled data was provided by Figure Eight.

## Execution

### Project Description

This project consists of three separate components:

1. ETL Pipeline

   Program file `data/process_data.py` merges disaster messages and labelled categories, cleans the data and exports the data into SQL database format.

2. ML Pipeline

   Program file `models/train_classifier.py` is responsible for learning the classification of disaster messages and export the learned model into a pickle file which can be used by the web app to make a prediction of an unseen message.

3. Web Application

   Program file `run.py` contains the Flask web application which allows user to enter a message and classify the disaster categories.

### Prerequisites

This program requires the following python packages:

- pandas
- SQLAlchemy
- numpy
- pickle
- NLTK
- sklearn
- json
- plotly
- flask
- joblib

### Execution Steps

1. Setting up database and model:

   At root directory,

   - To run ETL pipeline that cleans data and stores in database, use this command.

   `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

   - To run ML pipeline that trains classifier and saves the model, use this command.

   `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Running the web application:

   Within the `app`'s directory, use this command.

   `python run.py`

3. Launching the web application:

   Go to http://0.0.0.0:3001/

## Sample Screenshots

* Entering a message:

![Screenshot Entering Message](/images/message_input.png)

* Classification result:

![Screenshot of Classification Result](/images/classification_result.png)

* Training data visualization:

![Training data visualization 1](/images/training_visual1.png)
![Training data visualization 2](/images/training_visual2.png)
