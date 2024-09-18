import re
import urllib.parse
import cv2
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
from flask import Flask, request, Response, jsonify
from PIL import Image
import requests, io
from base64 import b64encode

app = Flask(__name__)

def load_sam_model():
    sam_checkpoint = "sam_vit_h_4b8939.pth"  
    model_type = "vit_h" 
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    predictor = SamPredictor(sam)
    return predictor


def extract_image_name(url):
    parsed_url = urllib.parse.urlparse(url)
    path = parsed_url.path
    filename = path.split('/')[-1]
    return filename


def parse_image_name(image_name):
    image_name=extract_image_name(image_name)
    base_name = re.sub(r"\.[a-zA-Z0-9]+$", "", image_name)
    print(f"Base name: {base_name}")
    
    parts = base_name.split(' - ')
    print(f"Parts after split: {parts}")
    
    if len(parts) != 2:
        raise ValueError(f"Filename does not match expected pattern: {base_name}")
    
    try:
        # Parsing the label
        label_part = parts[0].strip()
        if not label_part.isdigit():
            raise ValueError(f"Label part is not a number: {label_part}")
        input_label = np.ones(int(label_part), dtype=int)
        print(f"Input label: {input_label}")
        
        # Parsing the range
        range_str = parts[1].strip()
        ranges = []
        
        # Check if there are multiple ranges separated by '&'
        range_parts = range_str.split('&')
        for range_part in range_parts:
            start_end = range_part.split('-')
            if len(start_end) != 2:
                raise ValueError(f"Range part does not split into two integers: {range_part}")
            start, end = map(int, start_end)
            ranges.append([start, end])
        
        input_points = np.array(ranges)
        print(f"Input points: {input_points}")
        
        return input_label, input_points
    
    except ValueError as e:
        raise ValueError(f"Error parsing the image name: {base_name}") from e


def image_process(image_name, pattern):
    try:
        input_label, input_points = parse_image_name(image_name)
    except ValueError as e:
        print(e)
        return
    
    # Read the input image using OpenCV
    try:
        image = cv2.imread(image_name)
        if image is not None:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis('on')
            plt.show()
        else:
            print(f"Image not found: {image_name}")
            return
    except Exception as e:
        print(f"Error reading the image: {e}")
        return
    
    # Load SAM model and set the image
    predictor = load_sam_model()
    predictor.set_image(image)
    
    # Predict the mask using the SAM predictor
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_label,
        multimask_output=True
    )
    
    # Choose the best mask (highest score)
    best_mask = masks[np.argmax(scores)]
    
    # Read the new pattern image
    new_image = cv2.imread(pattern)
    if new_image is None:
        print(f"Pattern image not found: {pattern}")
        return
    
    # Convert the pattern image to RGB
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    
    # Process the mask and apply the new image to the region of interest
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x, y, w, h = cv2.boundingRect(best_mask.astype(np.uint8))
    resized_new_image = cv2.resize(new_image, (w, h))
    
    # Replace the masked region with the resized new image
    img[y:y+h, x:x+w][best_mask[y:y+h, x:x+w]] = resized_new_image[best_mask[y:y+h, x:x+w]]
    
    # Display the final result
    plt.imshow(img)
    plt.axis('on')
    plt.show()

@app.route('/process_images', methods=['POST'])
def process_images():
    product_image_url = request.json.get('product_image_url')
    room_image_url = request.json.get('room_image_url')

    if not product_image_url or not room_image_url:
        return 'Please provide product image URL and room image URL.', 400

    try:
        product_response = requests.get(product_image_url, stream=True)
        room_response = requests.get(room_image_url, stream=True)

        product_response.raise_for_status()
        room_response.raise_for_status()

        product_image = Image.open(io.BytesIO(product_response.content))
        room_image = Image.open(io.BytesIO(room_response.content))

        # Extract image names for logging or potential use
        product_image_name = extract_image_name(product_image_url)
        room_image_name = extract_image_name(room_image_url)

        # Process images using your image_process function

        processed_image = image_process(room_image, product_image)

        processed_image_buffer = io.BytesIO()
        processed_image.save(processed_image_buffer, format='JPEG')
        processed_image_data = processed_image_buffer.getvalue()

        base64_encoded_image = b64encode(processed_image_data).decode('utf-8')

        return jsonify({'processed_image': base64_encoded_image, 'format': 'JPEG'})

    except requests.exceptions.RequestException as e:
        print(f'Error fetching images: {e}')
        return 'Error fetching images.', 500
    except Exception as e:
        print(f'Error processing images: {e}')
        return 'Error processing images.', 500

if __name__ == '__main__':
  app.run(debug=True, port=5000)



