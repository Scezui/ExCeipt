import pytesseract
from flask import Flask, request, render_template, jsonify, send_file, redirect, url_for, flash, send_from_directory
from PIL import Image, ImageDraw
import torch
from transformers import LayoutLMv2ForTokenClassification, LayoutLMv3Tokenizer
import csv
import json
import subprocess
import os
import torch
import warnings
from PIL import Image
import sys
from fastai import *
from fastai.vision import *
from fastai.metrics import error_rate
from werkzeug.utils import secure_filename
import pandas as pd
from itertools import zip_longest
import inspect
from threading import Lock

import warnings

# Ignore SourceChangeWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore", category=SourceChangeWarning)


UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'supersecretkey'


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            resp = jsonify({'message' : 'No file part in the request'})
            resp.status_code = 400
            return resp
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('rename_file', old_name=filename))
    return render_template('index.html')


def make_prediction(image_path):
    try:
        # temp = pathlib.PosixPath  # Save the original state
        # pathlib.PosixPath = pathlib.WindowsPath  # Change to WindowsPath temporarily

        model_path = Path(r'model\export')

        learner = load_learner(model_path)

        # Open the image using fastai's open_image function
        image = open_image(image_path)

        # Make a prediction
        prediction_class, prediction_idx, probabilities = learner.predict(image)

        # If you want the predicted class as a string
        predicted_class_str = str(prediction_class)

        return predicted_class_str

    except Exception as e:
        return {"error": str(e)}
    # finally:
    #     pathlib.PosixPath = temp 


@app.route('/rename/<old_name>', methods=['GET', 'POST'])
def rename_file(old_name):
    new_name = 'temp.jpg'
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], old_name)
    new_file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_name)

    if os.path.exists(file_path):
        shutil.move(file_path, new_file_path)

        # Call make_prediction automatically
        prediction_result = make_prediction(new_file_path)

        return render_template('extractor.html', uploaded_file=new_name, old_name=old_name, prediction_result=prediction_result)
    else:
        return 'File not found'
    
    
@app.route('/get_inference_image')
def get_inference_image():
    # Assuming the new image is stored in the 'inferenced' folder with the name 'temp_inference.jpg'
    inferenced_image = 'inferenced/temp_inference.jpg'
    return jsonify(updatedImagePath=inferenced_image), 200  # Return the image path with a 200 status code
    
# Define a lock object
inference_lock = Lock()

@app.route('/run_inference', methods=['GET'])
def run_inference():
    # print(f"run_inference was called from {inspect.stack()[1].filename} at line {inspect.stack()[1].lineno}")
    if inference_lock.locked():
        return '', 204  # Return an empty response with a 204 status code
    
    # Acquire the lock before starting the inference process
    with inference_lock:
        try:

            model_path = r"model" # path to Layoutlmv3 model
            images_path = r"static/uploads" # images folder
            # Your inference process code here
            subprocess.check_call([sys.executable, "static/inference/run_inference.py", "--model_path", model_path, "--images_path", images_path])
            return redirect(url_for('create_csv'))
        except Exception as e:
            return jsonify({"error": str(e)})
            

# Define a function to replace all symbols with periods
def replace_symbols_with_period(value):
    return re.sub(r'\W+', '.', str(value))
@app.route('/create_csv', methods=['GET'])
def create_csv():
    try:
        # Load JSON data from file
        json_file_path = r"temp/LayoutlMV3InferenceOutput.json"  # path to JSON file
        output_file_path = r"inferenced/output.csv"  # path to output CSV file

        with open(json_file_path, 'r') as file:
            data = json.load(file)

        # Creating a dictionary to store labels and corresponding texts
        label_texts = {}
        for item in data:
            for output_item in item['output']:
                label = output_item['label']
                text = output_item['text']
                
                if label not in label_texts:
                    label_texts[label] = []
                label_texts[label].append(text)

        # Order of columns as requested
        column_order = [
            'RECEIPTNUMBER', 'MERCHANTNAME', 'MERCHANTADDRESS', 
            'TRANSACTIONDATE', 'TRANSACTIONTIME', 'ITEMS', 
            'PRICE', 'TOTAL', 'VATTAX'
        ]

        # Writing data to CSV file with ordered columns
        with open(output_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=column_order, delimiter="|")
            csv_writer.writeheader()

            # Iterating over each item and price, creating a new row for each pair
            items = label_texts.get('ITEMS', [])
            prices = label_texts.get('PRICE', [])
            
            for i in range(max(len(items), len(prices))):
                item_words = items[i].split() if i < len(items) else ['']
                price_words = prices[i].split() if i < len(prices) else ['']

                for j, (item, price) in enumerate(zip_longest(item_words, price_words, fillvalue='')):
                    row_data = {
                        'ITEMS': item,
                        'PRICE': replace_symbols_with_period(price) if 'PRICE' in label_texts else price  # Replace symbols with period
                    }
                    if j == 0:
                        row_data.update({
                            label: replace_symbols_with_period(label_texts[label][0]) if label in ['TOTAL', 'VATTAX'] and label in label_texts and 0 < len(label_texts[label]) else label_texts[label][0] if label in label_texts and 0 < len(label_texts[label]) else '' 
                            for label in column_order if label not in ['ITEMS', 'PRICE']
                        })
                    
                    csv_writer.writerow(row_data)

        return '', 204  # Return an empty response with a 204 status code
    except Exception as e:
        return jsonify({"error": str(e)})

        
@app.route('/get_data')
def get_data():
    return send_from_directory('inferenced','output.csv', as_attachment=False)

from flask import jsonify

@app.route('/download_csv', methods=['GET'])
def download_csv():
    try:
        output_file_path = r"inferenced\output.csv"  # path to output CSV file
        
        # Check if the file exists
        if os.path.exists(output_file_path):
            return send_file(output_file_path, as_attachment=True, download_name='output.csv')
        else:
            return jsonify({"error": "CSV file not found"})
    except Exception as e:
        return jsonify({"error": f"Download failed: {str(e)}"})


if __name__ == '__main__':
    app.run(debug=True)