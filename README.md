# DisasterResponse
This project tries to classify messages to different categories that would aid adequate response. The model was trained on disaster data from Figure Eight, which contains pre-labeled tweets and text messages from real-life disasters.

## File Descriptions
This project contains 3 folders.
* The data folder holds the data on which the model was trained. This is basically messages from real-life disasters alongside the label which shows the category they fall into.
* The models folder contains the python file for the model as well as the pickle file
* The app folder contains the flask app which makes use of the pickle file and displays a page where messages can be entered which in turn classifies the message into different categories

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

