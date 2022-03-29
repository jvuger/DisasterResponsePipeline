# DisasterResponsePipeline

In this project, disaster data from [Figure Eight](https://www.figure-eight.com) was analyzed in order to build a model that classifies disaster messages.

### Directory Structure

```bash
├── app
│   ├── templates
│   │   ├── go.html
│   │   └── master.html
│   └── run.py
├── data
│   ├── disaster_categories.csv
│   ├── disaster_messages.csv
│   ├── messag_categ.db
│   └── process_data.py
├── models
│   └── train_classifier.py
└── README.md
```


### Instructions:
1. Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Technologies
* Python 3.7.2

Libraries/modules used:
* json
* plotly
* pandas
* joblib
* nltk
* flask
* sqlalchemy
* numpy
* re
* sklearn


## Acknowledgements
Data provided by: [Udacity](https://www.udacity.com), [Figure Eight](https://www.figure-eight.com)

Files used: disaster_categories.csv, disaster_messages.csv
