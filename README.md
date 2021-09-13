# Disaster Response Pipeline Project

## Description

## Execution

### Prerequisites

This program requires the following python packages:

- pickle
- pandas
- numpy
- etc.   _To finish_


### Necessary Steps

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
