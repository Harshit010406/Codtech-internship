# End-to-End Data Science Project: Iris Classifier

This project demonstrates a full machine learning workflow from data preprocessing to deployment with Flask.

## Features
- Load Iris dataset
- Train RandomForestClassifier
- Save model using `joblib`
- Deploy with Flask and basic HTML frontend

## Setup
```bash
pip install -r requirements.txt
```

## Run the App
```bash
python model.py  # Train and save model
python app.py    # Launch Flask app
```

Visit `http://127.0.0.1:5000/` to use the web interface.
