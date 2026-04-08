import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """

    import json
    
    with open(info_path, 'r') as f:
        data = json.load(f)

    # Access the specific view (camera) data
    # In SuperTuxKart info files, 'views' is typically a list of camera perspectives
    # karts = data.get('karts', [])
    track = data.get('track', 'unknown_track')
    kart_names = data.get('karts', [])
    detections = data.get('detections',[])
    
    candidate_karts = []

    # 1. Setup Scaling (Match draw_detections logic)
    # Original STK resolution is 600x400
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT
    
    # Define the absolute center of the image frame
    img_center_x = img_width / 2
    img_center_y = img_height / 2
    
    for det in detections:
      for act_det in det:
    #     # Get the 2D center coordinates (x, y)
    #     # Note: Ensure these keys match the actual structure of your info.json
    #     class_id, obj_id, x1, y1, x2, y2 = act_det
    #     class_id = int(class_id)
    #     obj_id = int(obj_id)
    #     print("act_det",act_det)
    #     print("class,obj_ids:",class_id, obj_id)
    #     if class_id != 1:
    #         continue
    #     # Check if the kart's center is within the image frame
    #     raw_center_x = (x1 + x2) / 2
    #     raw_center_y = (y1 + y2) / 2

    #     # 2. Apply scaling factors 
    #     # (e.g., 150 / 600 = 0.25)
    #     scale_x = img_width / ORIGINAL_WIDTH
    #     scale_y = img_height / ORIGINAL_HEIGHT
    #     scaled_center_x = raw_center_x * scale_x
    #     scaled_center_y = raw_center_y * scale_y
    #     is_center_kart = False
    #     # 3. Check bounds against the RESIZED frame
    #     if 0 <= scaled_center_x < img_width and 0 <= scaled_center_y < img_height:
    #         name = kart_names[obj_id] if obj_id < len(kart_names) else f"kart_{kart_id}"
    #         if scaled_center_x == img_width/2 and scaled_center_y == img_height/2:
    #           is_center_kart = True
    #         print(obj_id,name,scaled_center_x,scaled_center_y,is_center_kart)  
    #         visible_karts.append({
    #             "instance_id": obj_id,
    #             "kart_name": name,
    #             "center": (scaled_center_x, scaled_center_y),
    #             "is_center_kart": is_center_kart
    #         })
            
            # return visible_karts
            obj_type, kart_id, x1, y1, x2, y2 = act_det
                
                # Only process Karts (Type 1)
            if obj_type == 1:
                # Calculate scaled center points (round to match pixel-based drawing if needed)
                # Center = (x1 + x2) / 2, then scaled
                cx = ((x1 + x2) / 2) * scale_x
                cy = ((y1 + y2) / 2) * scale_y
                    
                # Check if kart is in sight (within frame boundaries)
                if 0 <= cx < img_width and 0 <= cy < img_height:
                    # Calculate Euclidean distance to image center
                    dist = np.sqrt((cx - img_center_x)**2 + (cy - img_center_y)**2)
                        
                    candidate_karts.append({
                            "instance_id": kart_id,
                            "kart_name": kart_names[kart_id] if kart_id < len(kart_names) else f"kart_{kart_id}",
                            "center": (cx, cy),
                            "dist_to_center": dist,
                            "is_center_kart": False # Placeholder
                        })

    if not candidate_karts:
        return []

    # 3. Second Pass: Identify the Ego/Center Kart
    # The kart closest to the physical center of the image is the "ego"
    ego_kart = min(candidate_karts, key=lambda k: k["dist_to_center"])
    ego_kart["is_center_kart"] = True

    return candidate_karts


def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """

    import json
    
    with open(info_path, 'r') as f:
        data = json.load(f)

    # The track_id is a top-level key in the SuperTuxKart info.json files
    # common values include 'cocoa_temple', 'gran_paradiso_island', etc.
    track = data.get('track', 'unknown_track')
    
    return track


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of dictionaries, each containing a question and answer
    """
    # 1. Ego car question
    # What kart is the ego car?

    # 2. Total karts question
    # How many karts are there in the scenario?

    # 3. Track information questions
    # What track is this?

    # 4. Relative position questions for each kart
    # Is {kart_name} to the left or right of the ego car?
    # Is {kart_name} in front of or behind the ego car?
    # Where is {kart_name} relative to the ego car?

    # 5. Counting questions
    # How many karts are to the left of the ego car?
    # How many karts are to the right of the ego car?
    # How many karts are in front of the ego car?
    # How many karts are behind the ego car?

    questions = []
    
    # 1. Extract the data using your previously implemented functions
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_id = extract_track_info(info_path)
    
    # Find the ego car (the one the camera is attached to/centered on)
    ego_kart = next((k for k in karts if k['is_center_kart']), None)

    # --- 1. Ego car question ---
    if ego_kart:
        questions.append({
            "question": "What kart is the ego car?",
            "answer": ego_kart['kart_name']
        })

    # --- 2. Total karts question ---
    questions.append({
        "question": "How many karts are there in the scenario?",
        "answer": str(len(karts))
    })

    # --- 3. Track information question ---
    questions.append({
        "question": "What track is this?",
        "answer": track_id
    })

    # --- 4. Relative position & 5. Counting (Left/Right) ---
    if ego_kart:
        ego_x = ego_kart['center'][0]
        karts_to_the_left = 0
        karts_to_the_right = 0

        for kart in karts:
            if kart['is_center_kart']:
                continue
            
            kart_x = kart['center'][0]
            rel_pos = "left" if kart_x < ego_x else "right"
            
            # Question 4: Specific relative position
            questions.append({
                "question": f"Is {kart['kart_name']} to the left or right of the ego car?",
                "answer": rel_pos
            })
            
            if rel_pos == "left":
                karts_to_the_left += 1
            else:
                karts_to_the_right += 1

        # Question 5: Counting relative positions
        questions.append({
            "question": "How many karts are to the left of the ego car?",
            "answer": str(karts_to_the_left)
        })
        questions.append({
            "question": "How many karts are to the right of the ego car?",
            "answer": str(karts_to_the_right)
        })

    return questions


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.imsave("debug_output.png", annotated_image) 
    print("Image saved as debug_output.png - check the files sidebar!")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_qa_pairs})


if __name__ == "__main__":
    main()
