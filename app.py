import re
import urllib.parse
import cv2
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
from flask import Flask, request, jsonify
import requests, base64
from base64 import b64encode
import json

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
    
    parts = base_name.split('+-+')
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


def tile_image(replacement_img, target_shape, mask, scale_factor=0.2):
    # Resize the replacement image to make the tiles smaller
    small_tile = cv2.resize(replacement_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    tile_h, tile_w, _ = small_tile.shape
    target_h, target_w = target_shape[:2]

    # Create an empty target image of the same size as the room image
    tiled_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # Iterate over the target area and tile the replacement image
    for i in range(0, target_h, tile_h):
        for j in range(0, target_w, tile_w):
            # Determine where to place the current tile
            end_i = min(i + tile_h, target_h)
            end_j = min(j + tile_w, target_w)

            # Place the tile in the correct location, within the boundaries
            tiled_img[i:end_i, j:end_j] = small_tile[:end_i - i, :end_j - j]

    # Only apply the tiled image to the masked area
    return np.where(mask[:, :, None] > 0, tiled_img, 0)

def image_process(image_name, pattern, room_image_url):
    try:
        input_label, input_points = parse_image_name(room_image_url)
    except ValueError as e:
        print(e)
        return
    try:
        image = cv2.imdecode(image_name, cv2.IMREAD_COLOR)
        if image is not None:
            print("Image is loading")
        else:
            print(f"Image not found: {image_name}")
            return
    except Exception as e:
        print(f"Error reading the image: {e}")
        return
    
    # Load SAM model and set the image
    predictor = load_sam_model()
    predictor.set_image(image)
    

    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_label,
        multimask_output=True
    )

    best_mask = masks[np.argmax(scores)]
    
    # Read the new pattern image
    new_image = cv2.imdecode(pattern, cv2.IMREAD_COLOR)
    if new_image is None:
        if new_image is not None:
            print("Image is loading")
        else:
            print(f"Pattern image not found: {pattern}")
        return
    
    # Convert the pattern image to RGB
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    
    # Process the mask and apply the new image to the region of interest
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x, y, w, h = cv2.boundingRect(best_mask.astype(np.uint8))
    
    # Tile the image instead of simply resizing it
    mask_section = best_mask[y:y+h, x:x+w]
    tiled_image = tile_image(new_image, (h, w), mask_section)

    # Apply only to the masked area (pixel by pixel where mask is 1)
    mask_indices = np.where(mask_section)
    img[y:y+h, x:x+w][mask_indices] = tiled_image[mask_indices]
    return img
def numpy_to_base64(image_array):

    # Convert the image array to a BGR format (if necessary)
    if image_array.shape[-1] == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    _, encoded_image = cv2.imencode('.jpg', image_array)

    # Convert the encoded image to Base64
    base64_string = base64.b64encode(encoded_image).decode('utf-8')

    return base64_string

@app.route('/process_images', methods=['POST'])
def process_images():
    json_data=request.get_json()
    if json_data is None:
      return jsonify({'error': 'No JSON data provided'}), 400
    categoryname = json_data.get('categoryname')
    categoryname=categoryname.capitalize()
    image_urls = []
    for image in json_data.get('images', []):
      image_url = image.get('url')
      if image_url:
          image_urls.append(image_url)
    image_url_1=image_urls[0]
    image_url_2=image_urls[1]
    url=f"https://newbackend.ayatrio.com/api/fetchProductsByCategory/{categoryname}"
    response = requests.get(url)
    response.raise_for_status() 
    data=json.loads(response.text)
    product_image_url=data[0]['productImages'][0]['images'][0]
    try:
        product_response = requests.get(product_image_url, stream=True)
        room_response = requests.get(image_url_1, stream=True)

        product_response.raise_for_status()
        room_response.raise_for_status()
        product_image = np.frombuffer(product_response.content, np.uint8)
        room_image=np.frombuffer(room_response.content,np.uint8)

        processed_image = image_process(room_image, product_image,image_url_1)

        base64_encoded_image1=numpy_to_base64(processed_image)

        room_response = requests.get(image_url_2, stream=True)

        room_response.raise_for_status()
    
        room_image=np.frombuffer(room_response.content,np.uint8)

        processed_image = image_process(room_image, product_image,image_url_2)

        base64_encoded_image2=numpy_to_base64(processed_image)

        return jsonify({'processed_image1': base64_encoded_image1, 'processed_image2':base64_encoded_image2})

    except requests.exceptions.RequestException as e:
        print(f'Error fetching images: {e}')
        return 'Error fetching images.', 500
    except Exception as e:
        print(f'Error processing images: {e}')
        return 'Error processing images.', 500

if __name__ == '__main__':
  app.run(debug=True, port=5000)
