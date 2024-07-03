import pymongo
from flask import request
from pymongo.server_api import ServerApi
import dns.resolver
import certifi
import os

dns.resolver.default_resolver=dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers=['8.8.8.8']
#connect to mongo cluster using env variable MONGO_URI
client = pymongo.MongoClient(os.getenv('MONGO_URI'), server_api=ServerApi('1'),tlsCAFile=certifi.where())
userdb = client['userdb']
users = userdb.customers

def return_training_models():
	return userdb.training_models

def update_model_record(user_id, model_name, model_type, dataset_name, training_size, output_variable_name, status, rmse, accuracy=None):
    new_values = {"$set": {
        'user_id': user_id,
        'model_name': model_name,
        'model_type': model_type,
        'dataset_name': dataset_name,
        'training_size': training_size,
        'output_variable_name': output_variable_name,
        'status': status,
        'rmse': rmse,
        'accuracy': accuracy  # Include accuracy in the database update
    }}
    
    training_models.update_one({'model_name': model_name, 'user_id': user_id}, new_values, upsert=True)

def insert_data():
	if request.method == 'POST':
		name = request.form['name']
		email = request.form['email']
		password = request.form['pass']

		reg_user = {}
		reg_user['name'] = name
		reg_user['email'] = email
		reg_user['password'] = password

		if users.find_one({"email":email}) == None:
			users.insert_one(reg_user)
			return True
		else:
			return False


def check_user():

	if request.method == 'POST':
		email = request.form['email']
		password = request.form['pass']

		user = {
			"email": email,
			"password": password
		}

		user_data = users.find_one(user)
		if user_data == None:
			return False, ""
		else:
			return True, user_data["name"]
        

training_models = userdb.training_models

def insert_training_model(user_id, model_name, model_type, dataset_name, training_size, output_variable_name, input_vars):
    """
    Inserts a new training model record into the MongoDB collection.
    """
    training_model_record = {
        'user_id': user_id,
        'model_name': model_name,
        'model_type': model_type,
        'dataset_name': dataset_name,
        'training_size': training_size,
        'output_variable_name': output_variable_name,
        'input_vars': input_vars,
        'status': 'pending'
    }
    
    # Insert the record
    training_models.insert_one(training_model_record)

def get_training_models(user_id):
    """
    Retrieves training model records for a given user.
    """
    return list(training_models.find({'user_id': user_id}))