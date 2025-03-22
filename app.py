import os
import json
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    SqliteDatabase, PostgresqlDatabase, Model, IntegerField,
    FloatField, TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict


########################################
# Begin database stuff

DB = SqliteDatabase('predictions.db')


class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)

# End database stuff
########################################

########################################
# Unpickle the previously-trained model


with open(os.path.join('columns.json')) as fh:
    columns = json.load(fh)


with open(os.path.join('pipeline.pickle'), 'rb') as fh:
    pipeline = joblib.load(fh)


with open(os.path.join('dtypes.pickle'), 'rb') as fh:
    dtypes = pickle.load(fh)


# End model un-pickling
########################################


########################################
# Input validation functions


def check_id(request):
    """
        Validates that our request has observation_id
        
        Returns:
        - assertion value: True if request is ok, False otherwise
        - error message: empty if request is ok, False otherwise
    """
    
    if "observation_id" not in request:
        error = "Field `observation_id` missing from request: {}".format(request)
        return False, error
    
    return True, ""


def check_valid_column(observation):
    """
        Validates that our observation only has valid columns
        
        Returns:
        - assertion value: True if all provided columns are valid, False otherwise
        - error message: empty if all provided columns are valid, False otherwise
    """
    
    valid_columns = {
      "age",
      "sex",
      "race",
      "workclass",
      "education",
      "marital-status",
      "capital-gain",
      "capital-loss",
      "hours-per-week"
    }
    
    keys = set(observation.keys())
    
    if len(valid_columns - keys) > 0: 
        missing = valid_columns - keys
        error = "Missing columns: {}".format(missing)
        return False, error
    
    if len(keys - valid_columns) > 0: 
        extra = keys - valid_columns
        error = "Unrecognized columns provided: {}".format(extra)
        return False, error    

    return True, ""


def check_categorical_values(observation):
    """
        Validates that all categorical fields are in the observation and values are valid
        
        Returns:
        - assertion value: True if all provided categorical columns contain valid values, 
                           False otherwise
        - error message: empty if all provided columns are valid, False otherwise
    """
    
    valid_category_map = {
      "sex": ['Male', 'Female'],
      "race":["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"],
      "workclass":['Private', 'Self-emp-not-inc', 'Local-gov', '?',
            'State-gov', 'Self-emp-inc', 'Federal-gov', 'Without-pay', 'Never-worked'],
      "education":['HS-grad', 'Some-college', 'Bachelors', 'Masters', 'Assoc-voc',
            '11th', 'Assoc-acdm', '10th', '7th-8th', 'Prof-school', '9th', 'Doctorate', '12th',
            '5th-6th', '1st-4th', 'Preschool'],
      "marital-status":['Married-civ-spouse', 'Never-married', 'Divorced', 'Separated',
            'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'],
    }
    
    for key, valid_categories in valid_category_map.items():
        if key in observation:
            value = observation[key]
            if value not in valid_categories:
                error = "Invalid value provided for {}: {}. Allowed values are: {}".format(
                    key, value, ",".join(["'{}'".format(v) for v in valid_categories]))
                return False, error
        else:
            error = "Categorical field {} missing"
            return False, error

    return True, ""


def check_age(observation):
    """
        Validates that observation contains valid age value 
        
        Returns:
        - assertion value: True if age is valid, False otherwise
        - error message: empty if age is valid, False otherwise
    """
    
    age = observation.get("age")
        
    if not age:
        error = "Field `age` missing"
        return False, error

    if not isinstance(age, int):
        error = "Field `age` is not an integer"
        return False, error
    
    if age < 0 or age > 100:
        error = "Field `age` with {} is not between 0 and 100".format(age)
        return False, error

    return True, ""


def check_capital_gain(observation):
    """
        Validates that observation contains valid capital-gain value 
        
        Returns:
        - assertion value: True if capital-gain is valid, False otherwise
        - error message: empty if capital-gain is valid, False otherwise
    """
    
    capital_gain = observation.get("capital-gain") 
    
    if capital_gain is None: 
        error = "Field `capital-gain` missing"
        return False, error

    if not isinstance(capital_gain, int):
        error = "Field `capital-gain` is not an integer"
        return False, error
    
    if capital_gain < 0 or capital_gain > 99999:
        error = "Field `capital-gain` with {} is not between 0 and 99999".format(capital_gain)
        return False, error

    return True, ""

def check_capital_loss(observation):
    """
        Validates that observation contains valid hour value 
        
        Returns:
        - assertion value: True if hour is valid, False otherwise
        - error message: empty if hour is valid, False otherwise
    """

    capital_loss = observation.get("capital-loss")   
    
    if capital_loss is None: 
        error = "Field `capital-loss` missing"
        return False, error

    if not isinstance(capital_loss, int):
        error = "Field `capital-loss` is not an integer"
        return False, error
    
    if capital_loss < 0 or capital_loss > 4356:
        error = "Field `capital-loss` with {} is not between 0 and 4356".format(capital_loss)
        return False, error

    return True, ""

def check_hours(observation):
    """
        Validates that observation contains valid hour value 
        
        Returns:
        - assertion value: True if hour is valid, False otherwise
        - error message: empty if hour is valid, False otherwise
    """

    hours = observation.get("hours-per-week")   
    
    if hours is None: 
        error = "Field `hours-per-week` missing"
        return False, error

    if not isinstance(hours, int):
        error = "Field `hours-per-week` is not an integer"
        return False, error
    
    if hours < 0 or hours > 168:
        error = "Field `hours-per-week` with {} is not between 0 and 168".format(hours)
        return False, error

    return True, ""

# End input validation functions
########################################


########################################
# Begin webserver stuff

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    obs_dict = request.get_json()
    
    # If the `observation_id` is not present, set it to None.
    request_ok, error = check_id(obs_dict)
    if not request_ok:
        response = {'observation_id': None,
                    'error': error}
        return jsonify(response)
    else:
        _id = obs_dict['observation_id']
        
    try:
        observation = obs_dict['data']
    except:
        response = {'observation_id': _id,
                    'error': "data column is wrong or missing"}
        return jsonify(response)
    
    columns_ok, error = check_valid_column(observation)
    if not columns_ok:
        response = {'observation_id': _id, 'error': error}
        return jsonify(response)
    
    categories_ok, error = check_categorical_values(observation)
    if not categories_ok:
        response = {'observation_id': _id, 'error': error}
        return jsonify(response)
    
    age_ok, error = check_age(observation)
    if not age_ok:
        response = {'observation_id': _id, 'error': error}
        return jsonify(response)

    capital_gain_ok, error = check_capital_gain(observation)
    if not capital_gain_ok:
        response = {'observation_id': _id, 'error': error}
        return jsonify(response)

    capital_loss_ok, error = check_capital_loss(observation)
    if not capital_loss_ok:
        response = {'observation_id': _id, 'error': error}
        return jsonify(response)

    hours_ok, error = check_hours(observation)
    if not hours_ok:
        response = {'observation_id': _id, 'error': error}
        return jsonify(response)

    obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
    proba = pipeline.predict_proba(obs)[0, 1]
    prediction = pipeline.predict(obs)[0]
    response = {'observation_id': _id,
                'age': observation.get('age'),
                'sex': observation.get('sex'),
                'race': observation.get('race'),
                'workclass': observation.get('workclass'),
                'education': observation.get('education'),
                'marital-status': observation.get('marital-status'),
                'capital-gain': observation.get('capital-gain'),
                'capital-loss': observation.get('capital-loss'),
                'hours-per-week': observation.get('hours-per-week'),
                'prediction':  bool(prediction),
                'probability': proba}
    p = Prediction(
        observation_id=_id,
        proba=proba,
        observation=request.data,
    )
    p.save()
    return jsonify(response)

    
@app.route('/update', methods=['POST'])
def update():
    obs = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == obs['observation_id'])
        p.true_class = obs['true_class']
        p.save()
        return jsonify(model_to_dict(p))
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(obs['observation_id'])
        return jsonify({'error': error_msg})


    
if __name__ == "__main__":
    app.run()
