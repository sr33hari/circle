import db
from db import insert_training_model,update_model_record, return_training_models
import json
from flask import Flask, request, render_template, redirect, session, jsonify
# from flask_cors import CORS
import re
import os
import secrets
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from gspread.exceptions import SpreadsheetNotFound
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
import gspread #library for google sheets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import io
import logging

#Initialize Flask application and generate a random secret key to secure data in the application

app = Flask(__name__)
secret_key = secrets.token_hex(16)
# print(secret_key)
app.secret_key = secret_key

# cors = CORS(app) #This line may not be needed but in case of cors error, uncomment this

scope =  [
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/drive.file'
    ]

datasets_file_path = "datasets/dataset_info.json"
with open(datasets_file_path, 'r') as file:
        datasets_all_information= json.load(file)


@app.route('/update_datasets')
def update_datasets():
    user_name = session.get('username')
    if not user_name:
        return jsonify({'error': 'User not logged in or user_name not set in session'}), 401

    update_json_file_with_datasets(user_name)
    return jsonify({'success': True})

def update_json_file_with_datasets(user_name):
    # Load the existing JSON file
    with open(datasets_file_path, 'r') as json_file:
        data = json.load(json_file)
    
    # Assume this is the function that updates the 'data' dictionary
    updated_data = get_datasets_update(user_name, data)

    # Write the updated data back to the file
    with open(datasets_file_path, 'w') as json_file:
        json.dump(updated_data, json_file, indent=4)

def get_datasets_update(user_name, data):
    creds = ServiceAccountCredentials.from_json_keyfile_name("circle-418602-8aee1aeb2f00.json",scope)
    gc = gspread.authorize(creds)
    try: 
        sheet = gc.open('circle datasets').sheet1
    except SpreadsheetNotFound:
        sh = gc.create('circle datasets')
        sheet = sh.sheet1
    records = sheet.get_all_records()

    # Example logic to update the JSON structure based on your needs
    for record in records:
        if record['status(approved/not approved)'] == 'approved':
            # Download CSV, read it, and update data (details skipped)
            # Assuming dataset_name is unique and used as key
            dataset_name = record['name']
            # Placeholder for actual logic to get description, num_rows, num_columns, columns
            description = record['description']
            num_rows, num_columns, columns = download_google_sheet_as_csv(record['link'].split("id=")[1],user_name+"/"+dataset_name+".csv")
            
            # Update the JSON structure
            if user_name not in data['user_data']:
                data['user_data'][user_name] = {}
            data['user_data'][user_name][dataset_name] = {
                'description': description,
                'num_rows': num_rows,
                'num_columns': num_columns,
                'columns': columns
            }

    return data


def download_google_sheet_as_csv(file_id, local_file_path, credentials_json='circle-418602-8aee1aeb2f00.json'):
    """
    Exports a Google Sheets document as a CSV file.

    Parameters:
    - file_id: str. The ID of the Google Sheets document to download.
    - local_file_path: str. The full local file path where the CSV will be saved.
    - credentials_json: str. The path to the service account key file.
    """
    
    # Authenticate and create the Drive v3 API client
    credentials = service_account.Credentials.from_service_account_file(credentials_json, scopes=['https://www.googleapis.com/auth/drive.readonly'])
    service = build('drive', 'v3', credentials=credentials)
    
    # Ensure the local directory exists
    local_folder_path = local_file_path
    
    
    # Export the Google Sheets document as CSV
    request = service.files().export_media(fileId=file_id, mimeType='text/csv')
    with io.FileIO(local_folder_path, 'w') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            logging.debug(f"Download progress: {int(status.progress() * 100)}%")
            print(f"Download progress: {int(status.progress() * 100)}%")
    
    print(f"File has been downloaded to '{local_file_path}'")
    logging.info(f"File has been downloaded to '{local_file_path}'")
    df= pd.read_csv(local_folder_path)
    return len(df),len(df.columns),",".join(list(df.columns))



def get_confirm_token(response):
    """Parse the download confirmation token from Google Drive"""
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    """
    Save the content of the Google Drive file as specified in the destination.
    """
    CHUNK_SIZE = 32768  # 32KB chunks

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

@app.route('/upload', methods=['POST'])
def upload_file():
    # Upload file to Google Drive and get the link
    file = request.files['file']
    file_path = os.path.join('/tmp', file.filename)
    file.save(file_path)

    creds = ServiceAccountCredentials.from_json_keyfile_name("circle-418602-8aee1aeb2f00.json",scope)
    drive_service = build('drive', 'v3', credentials=creds)

    file_metadata = {
        'name': file.filename,
        'mimeType': 'application/vnd.google-apps.spreadsheet',
    }
    media = MediaFileUpload(file_path, mimetype='text/csv', resumable=True)
    file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    file_id = file.get('id')
    file_link = f'https://drive.google.com/open?id={file_id}'

    # Make the file public (optional)
    drive_service.permissions().create(fileId=file_id, body={'type': 'anyone', 'role': 'reader'}).execute()

    # Add details to Google Sheets
    gc = gspread.service_account(filename='circle-418602-8aee1aeb2f00.json')
    try:
        sheet = gc.open('circle datasets').sheet1
    except SpreadsheetNotFound:
        sh = gc.create('circle datasets')
        sheet = sh.sheet1

    # Assuming 'name', 'description' are the IDs of your form inputs
    name = request.form['name']
    description = request.form['description']
    user_name = session.get('username')
    sheet.append_row([user_name, name, description, file_link,"pending"])
    
    return jsonify({'success': True, 'link': file_link})        
        
@app.route('/', methods = ['GET', 'POST'])
def home():
    return render_template("signin.html")

@app.route('/home', methods = ['GET'])
def routeToMain():
    return render_template("index.html")

@app.route('/Datasets', methods = ['GET', 'POST'])
def datasets():
    with open(datasets_file_path, 'r') as file:
            datasets_all_information= json.load(file)
    datasets_all_information_to_send = datasets_all_information["global_data"] 
    if session.get('username') in list(datasets_all_information["user_data"].keys()):
        datasets_all_information_to_send = datasets_all_information["global_data"] | datasets_all_information["user_data"][session.get('username')]
        
    
    # print(datasets_all_information_to_send)
    logging.info("Datasets page loaded successfully")
    return render_template("Datasets.html", datasets_info = datasets_all_information_to_send,username = session.get('username'))

@app.route('/TrainModels', methods = ['GET', 'POST'])
def TrainModels():
    return render_template("TrainModels.html")


@app.route('/model_card')
def model_card():
    model_name = request.args.get('model_name')
    # Fetch model details from the database or another service
    user_id = session['username'] 
    models = list(training_models.find({"user_id": user_id}))
    m={}
    m["user_name"] = user_id
    for m in models:
        if m["model_name"] == model_name:
            model_details = m
    model_details["user_name"] = session['username'] 
    logging.info(f"Model card page loaded successfully for model: {model_name}")
    return render_template("model_card.html", model=model_details)


@app.route('/signup', methods = ['GET', 'POST'])
def signup():
    return render_template("signup.html")

@app.route('/logout')
def logout():
    # Remove 'username' from session or use session.clear() to remove everything
    session.pop('username', None)
    return redirect('/')

@app.route('/signin', methods = ['GET', 'POST'])
def signin():
    status, username = db.check_user()

    data = {
        "username": username,
        "status": status
    }
    if status:
        session['username'] = username
        # print(session['username'])
        logging.info(f"User {username} signed in successfully")
        return json.dumps(data)
    else:
        return json.dumps(data)

training_models = return_training_models()
print("Fetched training models from db\n\n\n\n ")

@app.route('/train_models_pipeline', methods=['POST'])
def train_models_pipeline():
    if 'username' not in session:
        return jsonify({'error': 'User not logged in'}), 401

    # Extract information from the request
    user_id = session['username']  # Assuming the username is the user_id
    request_data = request.json
    model_name = request_data.get('model_name')
    model_type = request_data.get('model_type')
    dataset_name = request_data.get('dataset_name')
    training_size = request_data.get('training_size')
    output_variable_name = request_data.get('output_variable_name')
    input_vars = request_data.get('input_variables')
    
    # Use the function from db.py to insert the training model information
    insert_training_model(user_id, model_name, model_type, dataset_name, training_size, output_variable_name, input_vars)
    
    try:
        process_pending_training_models()
    except Exception as e:
        print("exception in train model pipeline loop",e)
        
    return jsonify({'success': True, 'message': 'Training model pipeline initiated'})


@app.route('/get-dataset-columns')
def get_dataset_columns():
    dataset_name = request.args.get('dataset')
    print(dataset_name)
    # Assuming you have a function to load the dataset based on its name
    dataframe = load_dataset_as_dataframe(dataset_name)
    if dataframe is not None:
        columns = dataframe.columns.tolist()
        return jsonify({'columns': columns})
    else:
        return jsonify({'error': 'Dataset not found'}), 404

def load_dataset_as_dataframe(dataset_name):
    # Placeholder for loading the dataset as a DataFrame
    # You'll need to implement this based on how your datasets are stored
    # For example, if you have CSV files in a directory:
    with open(datasets_file_path, 'r') as file:
            datasets_all_information= json.load(file)
    datasets_all_information_to_send = datasets_all_information["global_data"] 
    if session.get('username') in list(datasets_all_information["user_data"].keys()) and dataset_name in list(datasets_all_information["user_data"][session.get('username')].keys()):
        path = session.get('username')+"/"+dataset_name+".csv"
    else:
        path = "datasets/"+dataset_name+".csv"
    # print(path)
    logging.debug(f"Loading dataset from path: {path}")
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None
    
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.metrics import r2_score

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from math import sqrt
import shap
import matplotlib.pyplot as plt

def train_model(model_name, model_type, dataset_name, output_variable_name, training_size,input_vars):
    
    input_vars = re.findall(r'<span class="tag">(.*?)<span', input_vars)
    
    df = load_dataset_as_dataframe(dataset_name)
    df = df.fillna(0)
    is_classification = df[output_variable_name].dtype == 'object' or df[output_variable_name].dtype.name == 'category'
    
    if is_classification:
        le = LabelEncoder()
        df[output_variable_name] = le.fit_transform(df[output_variable_name])
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    low_cardinality_cols = [col for col in categorical_cols if df[col].nunique() <= 10]
    high_cardinality_cols = list(set(categorical_cols) - set(low_cardinality_cols))
    #preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), low_cardinality_cols)], remainder='passthrough')
    
    if model_type == 'LinearRegression':
        model = LinearRegression()
        explainer_type = shap.LinearExplainer
    elif model_type == 'DecisionTree':
        model = DecisionTreeRegressor()
        explainer_type = shap.TreeExplainer
    elif model_type == 'RandomForest':
        model = RandomForestRegressor()
        explainer_type = shap.TreeExplainer
    elif model_type == 'XGBOOST':
        model = xgb.XGBRegressor()
        explainer_type = shap.TreeExplainer
    elif model_type == 'SVM':
        model = SVR()
        explainer_type = shap.KernelExplainer
    elif model_type == 'AdaBoost':
        model = AdaBoostRegressor()
        explainer_type = shap.KernelExplainer
    elif model_type == 'NeuralNetwork':
        # Assuming a simple neural network model from Keras
        from keras.models import Sequential
        from keras.layers import Dense
        model = Sequential([
            Dense(10, activation='relu', input_dim=len(numerical_cols) + len(low_cardinality_cols)),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        explainer_type = shap.DeepExplainer
    else:
        return "Error: Unsupported model type."
    df = df.select_dtypes(exclude=['object'])
    
    try:
        X = df.drop(columns=[output_variable_name])
    except:
        pass
    X = df[[col for col in input_vars if col in df.columns]]
    y = df[output_variable_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=int(training_size)/100, random_state=42)

    # Special case for neural networks
    if model_type == 'NeuralNetwork':
        model.fit(X_train, y_train, epochs=10, batch_size=10)
    else:
        #clf = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
    
    rmse = sqrt(mean_squared_error(y_test, predictions))
    accuracy = r2_score(y_test, predictions)
    
    print(f"Model {model_name} trained with RMSE: {rmse}, R2 Score: {accuracy}")
    logging.debug(f"Model {model_name} trained with RMSE: {rmse}, R2 Score: {accuracy}")
    update_model_record(session['username'], model_name, model_type, dataset_name, training_size, output_variable_name, 'completed', round(rmse, 2), round(accuracy, 2))
    
    # Generate and save SHAP plots
    imp_f_list = []
    if model_type != 'NeuralNetwork':
        imp_f_list = generate_and_save_shap_plots(model, X_train, model_name, explainer_type)
    else:
        imp_f_list = generate_and_save_shap_plots(model, X_train, model_name, shap.KernelExplainer)

    return "Model trained and evaluated successfully", imp_f_list
import shap
import matplotlib.pyplot as plt

def generate_and_save_shap_plots(model, X_train, model_id, explainer_type):
    np.bool = bool
    np.int = int
    np.float = float
    np.str = str
    user_name = session['username']
    directory_path = f'static/images/{user_name}'
    X_train = X_train.select_dtypes(exclude=['object'])
    os.makedirs(directory_path, exist_ok=True)
    def model_predict(data):
        return model.predict(data)
    flag = True
    # Initialize the appropriate SHAP explainer based on the explainer type
    if explainer_type == shap.Explainer:
        # General explainer, works for most model types including tree and linear models
        explainer = shap.Explainer(model.predict, X_train)
    elif explainer_type == shap.KernelExplainer:
        flag = False
        X_train = X_train.sample(10)
        explainer = explainer_type(model_predict, X_train)
        shap_values = explainer.shap_values(X_train,check_additivity=False)
        
    else:
        # Specific explainer for neural networks or complex models
        explainer = explainer_type(model, X_train)
    if flag:
    # Calculate SHAP values
        try:
            shap_values = explainer(X_train,check_additivity=False)
        except:
            shap_values = explainer(X_train)
    # Summary plot
    try:
        plt.figure()
        plt.tight_layout()
    
        shap.summary_plot(shap_values, X_train, show=False)
        plt.savefig(f'{directory_path}/shap_summary_{model_id}.png', pad_inches=0.3,bbox_inches='tight')
        plt.close()
    except Exception as e:
        print("exception in summary plot",e)
    try:
        # Dependence plot (plotting for the most impactful feature)
        try:
            feature_importance = np.abs(explainer.shap_values(X_train,check_additivity=False)).mean(0)
        except:
            feature_importance = np.abs(explainer.shap_values(X_train)).mean(0)
        most_impactful_feature = np.argmax(feature_importance)
        feature_list = list(X_train.columns)
        sorted_imp_features = np.argsort(feature_importance)
        imp_f_list = [feature_list[i] for i in sorted_imp_features][::-1]
        if len (imp_f_list ) > 4:
            imp_f_list = imp_f_list [:4]
        plt.figure()
        plt.tight_layout()
        shap.dependence_plot(most_impactful_feature, explainer.shap_values(X_train), X_train, show=False)
        
        plt.savefig(f'{directory_path}/shap_dependence_{model_id}.png', pad_inches=0.3,bbox_inches='tight')
        plt.close()
    except Exception as e:
        logging.error("exception in dependency plot",e)
        print("exception in dependency plot",e)
    # Bar plot
    try:
        plt.figure()
        plt.tight_layout()
    
        shap.plots.bar(shap_values[0], show=False)
        plt.savefig(f'{directory_path}/shap_bar_{model_id}.png', pad_inches=0.3,bbox_inches='tight')
        plt.close()
    except Exception as e:
        logging.error("exception in bar plot",e)
        print("exception in bar plot",e)
    try:
        # Waterfall plot for the first instance
        plt.figure()
        plt.tight_layout()
        
        shap.plots.waterfall(shap_values[0], show=False)
        plt.savefig(f'{directory_path}/shap_waterfall_{model_id}.png', pad_inches=0.3,bbox_inches='tight')
        plt.close()
    except Exception as e:
        logging.error("failed in waterfall plot",e)
        print("failed in waterfall plot",e)
    # Force plot (for multiple predictions)
    try:
        try:
            force_plot_html = shap.force_plot(explainer.expected_value, explainer.shap_values(X_train.sample(10),check_additivity=False), X_train.sample(10), show=False)
        except:
            force_plot_html = shap.force_plot(explainer.expected_value, explainer.shap_values(X_train.sample(10)), X_train.sample(10), show=False)

        shap.save_html(f'{directory_path}/force_plot_{model_id}.html', force_plot_html)
    except Exception as e:
        logging.error("failed in force plot",e)
        print("failed in force plot",e)
    return imp_f_list

def get_dataset_path(dataset_name):
    # print(dataset_name)
    logging.info(f"Fetching dataset path for dataset: {dataset_name}")
    # Assuming you have a function to load the dataset based on its name
    dataframe = load_dataset_as_dataframe(dataset_name)
    return dataframe
def process_pending_training_models():
    # Find all pending models
    pending_models = training_models.find({'status': 'pending'})
    # print(pending_models)
    logging.info(f"Processing pending training models for user: {session['username']}, fetching models {pending_models}")
    for model in pending_models:
        logging.info(f"Processing model {model}")
        # print(model)
        try:
            # Call the training function
            result, imp_f_list = train_model(
                model['model_name'],
                model['model_type'],
                # Assuming you have a mechanism to associate dataset names with paths
                model['dataset_name'],
                model['output_variable_name'],
                int(model['training_size']),
                model["input_vars"]
            )
            
            imp_f_list = ", ".join(imp_f_list)
            # Update the model status to 'completed' or similar
            training_models.update_one({'_id': model['_id']}, {'$set': {'status': 'completed',"imp_f_list":imp_f_list}})
            logging.info(f"Training completed for model: {model['model_name']}. Result: {result}")
            # print(f"Training completed for model: {model['model_name']}. Result: {result}")
        except Exception as e:
            # Optionally update the status to 'failed' and log the exception
            training_models.update_one({'_id': model['_id']}, {'$set': {'status': 'failed',"imp_f_list":" "}})
            logging.error(f"Error training model: {model['model_name']}. Exception: {e}")
            # print(f"Error training model: {model['model_name']}. Exception: {e}")

@app.route('/get-trained-models')
def get_trained_models():
    user_id = session.get('username')
    if not user_id:
        return jsonify({'error': 'User not logged in'}), 401

    models = list(training_models.find({"user_id": user_id}))[::-1]
    # Transform the models to be JSON serializable
    for model in models:
        model['_id'] = str(model['_id'])
    
    return jsonify(models)

@app.route('/get-datasets-for-user')
def get_datasets_for_user():
    user_name = session.get('username')
    if not user_name:
        return jsonify({'error': 'User not logged in'}), 401

    # Assuming you have a way to get datasets based on the user_name
    # This is a placeholder, implement according to your application logic
    datasets = get_datasets_by_user(user_name)
    
    return jsonify({'datasets': datasets})


@app.route('/Pretrained_Models', methods = ['GET', 'POST'])
def Pretrained_Models():
    return render_template("pre_trained.html")

@app.route('/Linear_mc', methods = ['GET', 'POST'])
def Linear_mc():
    return render_template("linear_mc.html")
@app.route('/XGBOOST_mc', methods = ['GET', 'POST'])
def XGBOOST_mc():
    return render_template("XGBOOST_mc.html")

@app.route('/Random_mc', methods = ['GET', 'POST'])
def Random_mc():
    return render_template("random_mc.html")

@app.route('/SVD_mc', methods = ['GET', 'POST'])
def SVD_mc():
    return render_template("svd_mc.html")

def get_datasets_by_user(user_name):
    with open(datasets_file_path, 'r') as file:
            datasets_all_information= json.load(file)
    datasets_all_information_to_send = datasets_all_information["global_data"] 
    if session.get('username') in list(datasets_all_information["user_data"].keys()):
        datasets_all_information_to_send = datasets_all_information["global_data"] | datasets_all_information["user_data"][session.get('username')]
        
    return  list(datasets_all_information_to_send.keys()) # Replace with actual data retrieval logic

@app.route('/register', methods = ['GET', 'POST'])
def register():
    status = db.insert_data()
    return json.dumps(status)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


if __name__ == "__main__":
    app.run()
 