import pandas as pd
import keras
import logging
import os
from sklearn.ensemble import RandomForestClassifier
from flask import jsonify
from google.cloud import storage
from sklearn import ensemble 

def train_titanic(train_data):
    y = train_data["Survived"]
    features = ["Pclass", "Sex", "SibSp", "Parch"]
    X = pd.get_dummies(train_data[features])
    X_test = pd.get_dummies(test_data[features])

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model.fit(X, y)
    predictions = model.predict(X_test)#
    
    scores = model.evaluate(X, y)
    print(model.metrics_names)
    text_out = {
        "accuracy:": scores[1],
        "loss": scores[0],}
    
    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
    output.to_csv('submission.csv', index=False)
    print("Your submission was successfully saved!")
    print(train_data_df)

    model_repo = os.environ['MODEL_REPO']
        if model_repo:
            file_path = os.path.join(model_repo, "model.h5")
            model.save(file_path)
            logging.info("Saved the model to the location : " + model_repo)
            return jsonify(text_out), 200
        else:
            model.save("model.h5")
            return jsonify({'message': 'The model was saved locally.'}), 200