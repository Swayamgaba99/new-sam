import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import SamPredictor, sam_model_registry


# Initialize the SAM model
def load_sam_model():
    sam_checkpoint = "sam_vit_h_4b8939.pth"  # Path to your SAM model checkpoint
    model_type = "vit_h"  # Model type: 'vit_h', 'vit_l', or 'vit_b'
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    predictor = SamPredictor(sam)
    return predictor

def parse_image_name(image_name):
    # Extract the base name without extension
    base_name = re.sub(r"\.[a-zA-Z0-9]+$", "", image_name)
    print(f"Base name: {base_name}")
    
    # Split the base name by ' - '
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

# Example usage
if __name__ == '__main__':
    image_name = r"C:\Users\anany\Desktop\house-ass\room\1 - 220-100.jpg"
    pattern = r"C:\Users\anany\Desktop\house-ass\room\wallpaper0.webp"
    
    # Run the function
    image_process(image_name, pattern)




