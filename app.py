import pytesseract
from flask import Flask, request, render_template, jsonify, send_file, redirect, url_for
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

        model_path = Path(r'model/export')

        learner = load_learner(model_path)

        # Open the image using fastai's open_image function
        image = open_image(image_path)

        # Make a prediction
        prediction_class, prediction_idx, probabilities = learner.predict(image)

        # If you want the predicted class as a string
        predicted_class_str = str(prediction_class)

        if predicted_class_str.lower() == 'receipt':
            # Redirect to run_inference route or any other route
            return '''This is a Valid Receipt
            '''
        else:
            return '''This is not a Valid Receipt
            '''   
            

    except Exception as e:
        return {"error": str(e)}
    # finally:
    #     pathlib.PosixPath = temp 

@app.route('/rename/<old_name>', methods=['GET', 'POST'])
def rename_file(old_name):
    new_name = 'temp.' + old_name.rsplit('.', 1)[-1]  # Get the file extension from the old_name
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], old_name)
    new_file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_name)

    if os.path.exists(file_path):
        shutil.move(file_path, new_file_path)

        # Call make_prediction automatically
        prediction_result = make_prediction(new_file_path)

        return render_template('extractor.html', uploaded_file=new_name, old_name=old_name, prediction_result=prediction_result)
    else:
        return 'File not found'

        
@app.route('/run_inference', methods=['GET'])
def run_inference():
    # defining inference parameters
    model_path =  Path(r'model') # path to Layoutlmv3 model
    images_path =  Path(r'static/uploads')# images folder
    # Build the command
    subprocess.check_call([sys.executable, "predictions/inference/run_inference.py", "--model_path", model_path, "--images_path", images_path])
    return redirect(url_for('create_csv'))


# Define a route for creating CSV
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

        with open(output_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=column_order)
            csv_writer.writeheader()
            
            # Determining the maximum number of texts for any label
            max_texts = max(len(label_texts[label]) if label in label_texts else 0 for label in column_order)
            
            # Filling in the CSV rows with texts for each label
            for i in range(max_texts):
                row_data = {
                    label: label_texts[label][i] if label in label_texts and i < len(label_texts[label]) else '' 
                    for label in column_order
                }
                
                # Splitting items and prices into new rows if they have more than one word
                if 'ITEMS' in row_data and 'PRICE' in row_data:
                    items = row_data['ITEMS'].split()
                    prices = row_data['PRICE'].split()
                    
                    if len(items) > 1 and len(prices) > 1:
                        # Write the first item and price in the first row
                        row_data['ITEMS'] = items[0]
                        row_data['PRICE'] = prices[0]
                        csv_writer.writerow(row_data)
                        
                        # Write the remaining items and prices in new rows
                        for item, price in zip(items[1:], prices[1:]):
                            csv_writer.writerow({'ITEMS': item, 'PRICE': price})

        return redirect(url_for('get_data'))
    except Exception as e:
        return jsonify({"error": str(e)})



'''@app.route('/download_csv', methods=['GET'])
def download_csv():
    try:
        output_file_path = r"inferenced\output.csv"  # path to output CSV file
        
        # Check if the file exists
        if os.path.exists(output_file_path):
            return send_file(output_file_path, as_attachment=True, download_name='output.csv')
        else:
            flash('CSV file not found', 'error')
            return render_template('extractor.html')
    except Exception as e:
        flash(f'Download failed: {str(e)}', 'error')
        return render_template('extractor.html')'''
    
@app.route('/get_data')
def get_data():
    # Adjust the path to your CSV file
    csv_path = 'inferenced/output.csv'
    
    # Send the CSV file as a response
    return send_file(csv_path, as_attachment=True)

@app.route('/update_data', methods=['POST'])
def update_data():
    try:
        data = request.json.get('data')
        
        # Path to the CSV file
        csv_path = 'inferenced/output.csv'

        # Write the updated data to the CSV file
        with open(csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            for row in data:
                csv_writer.writerow(row.split(','))

        return jsonify({'success': True, 'message': 'Data updated successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error updating data: {str(e)}'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
