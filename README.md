# Udacity-Disaster-Response-Pipeline
### Summary:
The goal of this project is to load the messages and categories data into a database, build a machine learning pipeline to classify the message text into its respective category, and then to deploy the model to a flask app. The app will display several visualizations, along with being able to classify new messages to its respective category.

### File Descriptions:
##### Data Folder
disaster_categories.csv - categories dataset
disaster_messages.csv - messages dataset
DisasterResponse.db - The database containing the merged and cleaned of the two data sets above
process_data.py - Python script that loads, cleans, and stores the data in the DisasterResponse database
##### Models Folder
train_classifier.py - Python script that models the data from DisasterResponse database
##### App Folder
templates - Contains html files for the flask web app
run.py - Runs the flask application
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
