from pathlib import Path

import fire
from matplotlib import pyplot as plt
import os
import glob
import json

from .generate_qa import draw_detections, extract_frame_info,extract_kart_objects,extract_track_info,get_spatial_and_count_info,get_image_filename


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    captions = []
    image_file = get_image_filename(info_path, view_index)
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    print("----karts:----",karts)
    track_id = extract_track_info(info_path)
    # Find the ego car (the one the camera is attached to/centered on)
    ego_kart = next((k for k in karts if k['is_center_kart']), None)

    if ego_kart:
        captions.append({
             "image_file": image_file,
             "caption": f"{ego_kart['kart_name']} is the ego car.",
        })
    captions.append({
             "image_file": image_file,
             "caption": f"The track is {track_id}.",
        })
     # Get data from our helper
    if len(karts) > 0:
     spatial_data, counts = get_spatial_and_count_info(karts, ego_kart)
     captions.append({
             "image_file": image_file,
             "caption": f"There are {len(karts)} karts in the scene.",
        })
     kart_names = [obj['name'] for obj in spatial_data] 
     captions.append({
             "image_file": image_file,
             "caption": f"The karts in the scene are {kart_names}.",
        })


     for data in spatial_data:
        name = data['name']
        v_rel = data['v_rel']
        h_rel = data['h_rel']
        captions.append({
             "image_file": image_file,
             "caption": f"{name} is {h_rel} of the ego car.",
        })
        if v_rel == 'front':
            captions.append({
             "image_file": image_file,
             "caption": f"{name} is in {v_rel} of the ego car.",
        })
        elif v_rel == 'back':
             captions.append({
             "image_file": image_file,
             "caption": f"{name} is {v_rel} the ego car.",
        })


    # 1. Ego car
    # {kart_name} is the ego car.

    # 2. Counting
    # There are {num_karts} karts in the scenario.

    # 3. Track name
    # The track is {track_name}.

    # 4. Relative position left or right
    # {kart_name} is {position} of the ego car.

    # 5. Relative position front or behind
    # {kart_name} is {position} of the ego car.

    # 6. count of karts
    # The karts in the scene are {kart_names}.

    return captions

def generate_captions_all(data_dir: str = 'data/train'):
    """
    Iterates through all info.json files in the directory and calls 
    generate_qa_pairs for each of the 10 kart views.
    """ 
    all_captions_pairs = []
    
    # 1. Find all info files in the specified directory
    # Using sorted() ensures the images are processed in numerical order
    info_files = sorted(glob.glob(os.path.join(data_dir, "*_info.json")))
    
    print(f"Found {len(info_files)} info collections. Processing 10 views per file...")

    # 2. Loop through each collection
    for info_path in info_files:
        
        # 3. Each collection contains data for 10 karts (view_index 0-9)
        for view_idx in range(10):
            
            # 4. Call generate_qa_pairs with the specified signature
            # This function now handles extraction, spatial logic, and image_file naming
            caption_list = generate_caption(
                info_path=info_path, 
                view_index=view_idx, 
                img_width=150, 
                img_height=100
            )
            
            # 5. Append the resulting list of pairs if it's not empty
            if caption_list:
                all_captions_pairs.extend(caption_list)
    print("count of captions pairs:",len(all_captions_pairs))
    # 6. Define the target output path
    output_path = os.path.join(data_dir, 'generated_captions.json')
    
    # 7. Write the complete list to the JSON file
    with open(output_path, 'w') as f:
        json.dump(all_captions_pairs, f, indent=2)
    
    print(f"Done! Created {len(all_captions_pairs)} total records.")
    print(f"File saved to: {output_path}")

    return all_captions_pairs 


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_caption,"generate_caption":generate_caption,"generate_caption_all":generate_captions_all})


if __name__ == "__main__":
    main()
