import os
import pandas as pd
import cv2
import numpy as np

def preprocess_image(image_path):
  try:
    image = cv2.imread(image_path)

    # Enhance text
    enhanced = enhance_txt(image)

    return enhanced

  except Exception as e:
    print(f"An error occurred: {str(e)}")
    return None


def enhance_txt(img, intensity_increase=20, bilateral_filter_diameter=9, bilateral_filter_sigma_color=75, bilateral_filter_sigma_space=75):
    w = img.shape[1]
    h = img.shape[0]
    w1 = int(w * 0.05)
    w2 = int(w * 0.95)
    h1 = int(h * 0.05)
    h2 = int(h * 0.95)
    ROI = img[h1:h2, w1:w2]  # 95% of the center of the image
    threshold = np.mean(ROI) * 0.98  # % of average brightness

    blurred = cv2.GaussianBlur(img, (1, 1), 0)
    edged = 255 - cv2.Canny(blurred, 100, 150, apertureSize=7)

    # Increase intensity by adding a constant value
    img = np.clip(img + intensity_increase, 0, 255).astype(np.uint8)

    # Apply bilateral filter to reduce noise
    img = cv2.bilateralFilter(img, bilateral_filter_diameter, bilateral_filter_sigma_color, bilateral_filter_sigma_space)

    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    return binary


def run_tesseract_on_preprocessed_image(preprocessed_image, image_path):  # -> tsv output path
    image_name = os.path.basename(image_path)
    image_name = image_name[:image_name.find('.')]

    # Create the "temp" folder if it doesn't exist
    temp_folder = "temp"
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    # Save the preprocessed image (optional)
    cv2.imwrite(f"{temp_folder}/{image_name}_preprocessed.png", preprocessed_image)

    # Run Tesseract on the preprocessed image
    error_code = os.system(f'''
        tesseract "{temp_folder}/{image_name}_preprocessed.png" "{temp_folder}/{image_name}" -l eng tsv
    ''')

    if not error_code:
        print("Saved TSV files")
        return f"{temp_folder}/{image_name}.tsv"
    else:
        raise ValueError('Tesseract OCR Error. Please verify image format (PNG, JPG, JPEG)')


def clean_tesseract_output(tsv_output_path):
  ocr_df = pd.read_csv(tsv_output_path, sep='\t')
  ocr_df = ocr_df.dropna()
  ocr_df = ocr_df.drop(ocr_df[ocr_df.text.str.strip() == ''].index)
  text_output = ' '.join(ocr_df.text.tolist())
  words = []
  for index, row in ocr_df.iterrows():
    word = {}
    origin_box = [row['left'], row['top'], row['left'] +
                  row['width'], row['top']+row['height']]
    word['word_text'] = row['text']
    word['word_box'] = origin_box
    words.append(word)
    # print(f"Cleaning Tesseract output: {tsv_output_path}")

  return words


def prepare_batch_for_inference(image_paths):
  print("Preparing for Inference")
  tsv_output_paths = []

  inference_batch = dict()
  for image_path in image_paths:
    preprocessed_image = preprocess_image(image_path)
    tsv_output_path = run_tesseract_on_preprocessed_image(preprocessed_image, image_path)
    tsv_output_paths.append(tsv_output_path)

  # clean_outputs is a list of lists
  clean_outputs = [clean_tesseract_output(tsv_path) for tsv_path in tsv_output_paths]
  word_lists = [[word['word_text'] for word in clean_output] for clean_output in clean_outputs]
  boxes_lists = [[word['word_box'] for word in clean_output] for clean_output in clean_outputs]

  inference_batch = {
    "image_path": image_paths,
    "bboxes": boxes_lists,
    "words": word_lists
  }

  print("Prepared for Inference Batch")
  return inference_batch