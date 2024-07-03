from flask import Blueprint, request, session, jsonify
import json
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from gspread.exceptions import SpreadsheetNotFound
from google.oauth2 import service_account
import pandas as pd
import logging
import io

api_blueprint = Blueprint('api_blueprint', __name__)
scope = [
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/drive.file'
]

datasets_file_path = "datasets/dataset_info.json"

@api_blueprint.route('/update_datasets')
def update_datasets():
    user_name = session.get('username')
    if not user_name:
        return jsonify({'error': 'User not logged in or user_name not set in session'}), 401

    update_json_file_with_datasets(user_name)
    return jsonify({'success': True})

def update_json_file_with_datasets(user_name):
    with open(datasets_file_path, 'r') as json_file:
        data = json.load(json_file)
    updated_data = get_datasets_update(user_name, data)
    with open(datasets_file_path, 'w') as json_file:
        json.dump(updated_data, json_file, indent=4)

def get_datasets_update(user_name, data):
    creds = ServiceAccountCredentials.from_json_keyfile_name("circle-418602-8aee1aeb2f00.json", scope)
    gc = gspread.authorize(creds)
    try:
        sheet = gc.open('circle datasets').sheet1
    except SpreadsheetNotFound:
        sh = gc.create('circle datasets')
        sheet = sh.sheet1
    records = sheet.get_all_records()
    for record in records:
        if record['status(approved/not approved)'] == 'approved':
            dataset_name = record['name']
            description = record['description']
            num_rows, num_columns, columns = download_google_sheet_as_csv(record['link'].split("id=")[1], user_name + "/" + dataset_name + ".csv")
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
    credentials = service_account.Credentials.from_service_account_file(credentials_json, scopes=['https://www.googleapis.com/auth/drive.readonly'])
    service = build('drive', 'v3', credentials=credentials)
    local_folder_path = local_file_path
    request = service.files().export_media(fileId=file_id, mimeType='text/csv')
    with io.FileIO(local_folder_path, 'w') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            logging.debug(f"Download progress: {int(status.progress() * 100)}%")
    df = pd.read_csv(local_folder_path)
    return len(df), len(df.columns), ",".join(list(df.columns))

@api_blueprint.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    file_path = os.path.join('/tmp', file.filename)
    file.save(file_path)
    creds = ServiceAccountCredentials.from_json_keyfile_name("circle-418602-8aee1aeb2f00.json", scope)
    drive_service = build('drive', 'v3', credentials=creds)
    file_metadata = {
        'name': file.filename,
        'mimeType': 'application/vnd.google-apps.spreadsheet',
    }
    media = MediaFileUpload(file_path, mimetype='text/csv', resumable=True)
    file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    file_id = file.get('id')
    file_link = f'https://drive.google.com/open?id={file_id}'
    drive_service.permissions().create(fileId=file_id, body={'type': 'anyone', 'role': 'reader'}).execute()
    gc = gspread.service_account(filename='circle-418602-8aee1aeb2f00.json')
    try:
        sheet = gc.open('circle datasets').sheet1
    except SpreadsheetNotFound:
        sh = gc.create('circle datasets')
        sheet = sh.sheet1
    name = request.form['name']
    description = request.form['description']
    user_name = session.get('username')
    sheet.append_row([user_name, name, description, file_link, "pending"])
    
    return jsonify({'success': True, 'link': file_link})

@api_blueprint.route('/get-dataset-columns')
def get_dataset_columns():
    dataset_name = request.args.get('dataset')
    dataframe = load_dataset_as_dataframe(dataset_name)
    if dataframe is not None:
        columns = dataframe.columns.tolist()
        return jsonify({'columns': columns})
    else:
        return jsonify({'error': 'Dataset not found'}), 404

def load_dataset_as_dataframe(dataset_name):
    with open(datasets_file_path, 'r') as file:
        datasets_all_information = json.load(file)
    datasets_all_information_to_send = datasets_all_information["global_data"]
    if session.get('username') in datasets_all_information["user_data"].keys() and dataset_name in datasets_all_information["user_data"][session.get('username')].keys():
        path = session.get('username') + "/" + dataset_name + ".csv"
    else:
        path = "datasets/" + dataset_name + ".csv"
    logging.debug(f"Loading dataset from path: {path}")
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None

@api_blueprint.route('/get-trained-models')
def get_trained_models():
    user_id = session.get('username')
    if not user_id:
        return jsonify({'error': 'User not logged in'}), 401
    models = list(db.return_training_models().find({"user_id": user_id}))[::-1]
    for model in models:
        model['_id'] = str(model['_id'])
    return jsonify(models)

@api_blueprint.route('/get-datasets-for-user')
def get_datasets_for_user():
    user_name = session.get('username')
    if not user_name:
        return jsonify({'error': 'User not logged in'}), 401
    datasets = get_datasets_by_user(user_name)
    return jsonify({'datasets': datasets})

def get_datasets_by_user(user_name):
    with open(datasets_file_path, 'r') as file:
        datasets_all_information = json.load(file)
    datasets_all_information_to_send = datasets_all_information["global_data"]
    if session.get('username') in datasets_all_information["user_data"].keys():
        datasets_all_information_to_send = {**datasets_all_information["global_data"], **datasets_all_information["user_data"][session.get('username')]}
    return list(datasets_all_information_to_send.keys())