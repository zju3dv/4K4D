import os
from easyvolcap.utils.console_utils import *

def generate_gif_from_pngs(directory):
    for root, dirs, files in os.walk(directory):
        # Check if the directory is not empty
        if not files:
            continue

        if all(f"{i:04d}.png" in files for i in range(len(files))):
            gif_name = os.path.basename(root) + ".gif"
            
            # Define the path for the gif to be at the same level as root
            gif_path = os.path.join(root, '..', gif_name)
            png_str = os.path.join(root, "%04d.png")

            # Construct the ffmpeg command
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-framerate", "30",
                "-f", "image2", "-nostdin", "-y", "-r", "30", "-i",
                f'"{png_str}"', "-lavfi",
                "palettegen=stats_mode=single[pal],[0:v][pal]paletteuse=new=1", gif_path
            ]

            # Execute the command
            run(cmd)
            print(f"Generated GIF for directory: {root}")


# Directory path
DIRECTORY_PATH = 'data/output'  # Replace with your directory path
generate_gif_from_pngs(DIRECTORY_PATH)
