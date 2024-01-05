import os
import pandas as pd
import cv2
import numpy as np
import json
import requests
import traceback

from PIL import Image

def preprocess_image(image_path, max_file_size_mb=1, target_file_size_mb=0.5):
    try:
        # Check file size
        file_size_mb = os.path.getsize(image_path) / (1024 * 1024)  # Convert to megabytes
        if file_size_mb > max_file_size_mb:
            print(f"File size ({file_size_mb} MB) exceeds the maximum allowed size ({max_file_size_mb} MB). Resizing the image.")

            # Read the image
            image = cv2.imread(image_path)

            # Calculate the new dimensions to achieve the target file size
            ratio = target_file_size_mb / file_size_mb
            new_width = int(image.shape[1] * np.sqrt(ratio))
            new_height = int(image.shape[0] * np.sqrt(ratio))
            
            # Enhance text
            enhanced_img = enhance_txt(image)
            
            # Resize the image
            enhanced = cv2.resize(enhanced_img, (new_width, new_height))

            return enhanced

        else:
            # If the file size is within the limit, proceed with the regular enhancement
            image = cv2.imread(image_path)
            enhanced = enhance_txt(image)
            return enhanced

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


def enhance_txt(img, intensity_increase=20, bilateral_filter_diameter=9, bilateral_filter_sigma_color=75, bilateral_filter_sigma_space=75):
    # Get the width and height of the image
    w = img.shape[1]
    h = img.shape[0]
    w1 = int(w * 0.05)
    w2 = int(w * 0.95)
    h1 = int(h * 0.05)
    h2 = int(h * 0.95)
    ROI = img[h1:h2, w1:w2]  # 95% of the center of the image
    threshold = np.mean(ROI) * 0.88  # % of average brightness

    blurred = cv2.GaussianBlur(img, (1, 1), 0)
    edged = 255 - cv2.Canny(blurred, 100, 150, apertureSize=7)

    # Increase intensity by adding a constant value
    img = np.clip(img + intensity_increase, 0, 255).astype(np.uint8)

    # Apply bilateral filter to reduce noise
    img = cv2.bilateralFilter(img, bilateral_filter_diameter, bilateral_filter_sigma_color, bilateral_filter_sigma_space)

    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    return binary


def run_tesseract_on_preprocessed_image(preprocessed_image, image_path):
    try:
        image_name = os.path.basename(image_path)
        image_name = image_name[:image_name.find('.')]

        # Create the "temp" folder if it doesn't exist
        temp_folder = "temp"
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)

        # Define the OCR API endpoint
        url = "https://api.ocr.space/parse/image"

        # Define the API key and the language
        api_key = "K88232854988957"  # Replace with your actual OCR Space API key
        language = "eng"

        # Save the preprocessed image
        cv2.imwrite(os.path.join(temp_folder, f"{image_name}_preprocessed.jpg"), preprocessed_image)

        # Open the preprocessed image file as binary
        with open(os.path.join(temp_folder, f"{image_name}_preprocessed.jpg"), "rb") as f:
            # Define the payload for the API request
            payload = {
                "apikey": api_key,
                "language": language,
                "isOverlayRequired": True,
                "OCREngine": 2
            }
            # Define the file parameter for the API request
            file = {
                "file": f
            }
            # Send the POST request to the OCR API
            response = requests.post(url, data=payload, files=file)

        # Check the status code of the response
        if response.status_code == 200:
            # Parse the JSON response
            result = response.json()
            print("---JSON file saved")
            # Save the OCR result as JSON
            with open(os.path.join(temp_folder, f"{image_name}_ocr.json"), 'w') as f:
                json.dump(result, f)

            return os.path.join(temp_folder, f"{image_name}_ocr.json")
        else:
            # Print the error message
            print("Error: " + response.text)
            return None

    except Exception as e:
        print(f"An error occurred during OCR request: {str(e)}")
        return None

def clean_tesseract_output(json_output_path):
    try:
        with open(json_output_path, 'r') as json_file:
            data = json.load(json_file)

        lines = data['ParsedResults'][0]['TextOverlay']['Lines']

        words = []
        for line in lines:
            for word_info in line['Words']:
                word = {}
                origin_box = [
                    word_info['Left'],
                    word_info['Top'],
                    word_info['Left'] + word_info['Width'],
                    word_info['Top'] + word_info['Height']
                ]

                word['word_text'] = word_info['WordText']
                word['word_box'] = origin_box
                words.append(word)

        return words
    except (KeyError, IndexError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error cleaning Tesseract output: {str(e)}")
        return None

def prepare_batch_for_inference(image_paths):
    # print("my_function was called")
    # traceback.print_stack()  # This will print the stack trace
    print(f"Number of images to process: {len(image_paths)}")  # Print the total number of images to be processed
    print("1. Preparing for Inference")
    tsv_output_paths = []

    inference_batch = dict()
    print("2. Starting Preprocessing")
    # Ensure that the image is only 1
    for image_path in image_paths:
        print(f"Processing the image: {image_path}")  # Print the image being processed
        print("3. Preprocessing the Receipt")
        preprocessed_image = preprocess_image(image_path)
        if preprocessed_image is not None:
            print("4. Preprocessing done. Running OCR")
            json_output_path = run_tesseract_on_preprocessed_image(preprocessed_image, image_path)
            print("5. OCR Complete")
            if json_output_path:
                tsv_output_paths.append(json_output_path)
                
    print("6. Preprocessing and OCR Done")
    # clean_outputs is a list of lists
    clean_outputs = [clean_tesseract_output(tsv_path) for tsv_path in tsv_output_paths]
    print("7. Cleaned OCR output")
    word_lists = [[word['word_text'] for word in clean_output] for clean_output in clean_outputs]
    print("8. Word List Created")
    boxes_lists = [[word['word_box'] for word in clean_output] for clean_output in clean_outputs]
    print("9. Box List Created")
    inference_batch = {
        "image_path": image_paths,
        "bboxes": boxes_lists,
        "words": word_lists
    }

    print("10. Prepared for Inference Batch")
    return inference_batch