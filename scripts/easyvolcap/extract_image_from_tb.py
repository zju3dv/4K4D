import os
from easyvolcap.utils.console_utils import *
from tensorboard.backend.event_processing import event_accumulator
from PIL import Image
from io import BytesIO

# Configuration parameters
log_dir = 'data/record/easyvolcap'
output_dir = 'data/output'
cameras_to_extract = ['camera0000', 'camera0005', 'camera0010', 'camera0015', 'camera0020']  # Specify the cameras you want to extract
frames_to_extract = ['frame0000', 'frame0005', 'frame0009']
types_to_extract = ['RENDER', 'DEPTH']

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Extract images from TensorBoard logs
ea = event_accumulator.EventAccumulator(log_dir, size_guidance={'images': 0})
ea.Reload()


pbar = tqdm(total=len(cameras_to_extract) * len(frames_to_extract) * len(types_to_extract))
for tag in ea.Tags()['images']:
    if 'VAL_FRAME' in tag:
        for camera in cameras_to_extract:
            for frame in frames_to_extract:
                for t in types_to_extract:
                    if camera in tag and frame in tag and t in tag and 'gt' not in tag and 'error' not in tag:
                        img_events = ea.Images(tag)
                        for index, img_event in enumerate(img_events[::10]):
                            img_data = BytesIO(img_event.encoded_image_string)
                            img = Image.open(img_data)
                            img_name = f"{t}/{camera}_{frame}/{index:04d}.png"
                            img_path = os.path.join(output_dir, img_name)
                            os.makedirs(os.path.dirname(img_path), exist_ok=True)
                            img.save(img_path)
                    pbar.update()
pbar.close()
print("Finished extracting images!")
