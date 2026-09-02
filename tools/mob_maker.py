import requests
import json
import zipfile
import io
import cv2
import numpy as np
from pathlib import Path

# Assuming a base URL for the API. In a real application, this would be in a config file.
BASE_URL = "https://maplestory.io" # Placeholder URL, replace with actual API base URL

# Default region and version if not provided
DEFAULT_REGION = "GMS"
DEFAULT_VERSION = "65"

def get_all_mobs(region=None, version=None):
    """
    Fetches world map data from the MapleStory API.

    Args:
        region (str, optional): The region of the map (e.g., "KMS", "GMS"). Defaults to DEFAULT_REGION.
        version (str, optional): The API version. Defaults to DEFAULT_VERSION.

    Returns:
        dict or None: A dictionary containing the world map data if successful, None otherwise.
    """
    if region is None:
        region = DEFAULT_REGION
    if version is None:
        version = DEFAULT_VERSION

    url = f"{BASE_URL}/api/{region}/{version}/mob"
    print(f"Fetching mobs from: {url}\nYou can find monster names at https://maplestory.wiki/GMS/65/mob")

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.json()
    
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error occurred: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"An unexpected error occurred: {req_err}")
    except json.JSONDecodeError:
        print(f"Failed to decode JSON from response: {response.text}")
    return None

def find_mob_id(all_mobs, mob_name):
    """
    Finds a monster by name in all mobs data.
    """
    mob_name_lower = mob_name.lower() # Convert input to lowercase for case-insensitive comparison
    result = next((mob for mob in all_mobs if mob["name"].lower() == mob_name_lower), None)
    return result['id'] if result else None

def save_mob(mob_id, folder="monster", mob_name="mob"):
    # Create output directory path: ./folder/mob_name
    output_dir = Path(".") / folder / mob_name
    # Create the directory and all parent directories if they don't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Construct the URL to download the mob zip file
    download_url = f"{BASE_URL}/api/{DEFAULT_REGION}/{DEFAULT_VERSION}/mob/{mob_id}/download"

    try:
        # Send HTTP GET request to download the zip file content
        response = requests.get(download_url)
        # Raise exception if HTTP request returned an unsuccessful status code
        response.raise_for_status()

        # Create a BytesIO stream from the downloaded zip file bytes (in-memory file)
        zip_bytes = io.BytesIO(response.content)

        # Open the zip file from the in-memory bytes
        with zipfile.ZipFile(zip_bytes) as zip_file:
            index = 1  # Counter to name saved images sequentially

            # Iterate over all files inside the zip archive
            for file_name in zip_file.namelist():
                # Skip files that contain "die1" in their name (case insensitive)
                if "die1" in file_name.lower():
                    continue

                # Open each image file inside the zip
                with zip_file.open(file_name) as file:
                    # Read the image data as bytes
                    image_data = file.read()
                    # Convert byte data to a NumPy array for OpenCV
                    np_arr = np.frombuffer(image_data, np.uint8)
                    # Decode the image data into an OpenCV image matrix, preserving alpha channel if present
                    img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

                    # If image decoding fails, print a warning and skip this file
                    if img is None:
                        print(f"Failed to decode image: {file_name}")
                        continue

                    # If the image has an alpha channel (4 channels)
                    if img.shape[2] == 4:
                        # Extract the alpha channel
                        alpha_channel = img[:, :, 3]
                        # Find pixels that are fully transparent (alpha == 0)
                        transparent_pixels = (alpha_channel == 0)
                        # Replace transparent pixels with solid green color (BGR: 0,255,0) and full opacity (255)
                        img[transparent_pixels, 0] = 0    # Blue channel
                        img[transparent_pixels, 1] = 255  # Green channel
                        img[transparent_pixels, 2] = 0    # Red channel
                        img[transparent_pixels, 3] = 255  # Alpha channel (opaque)

                    # Prepare the filename for saving the processed image
                    new_filename = f"{mob_name}_{index}.png"
                    save_path = output_dir / new_filename
                    # Save the processed image to disk
                    cv2.imwrite(str(save_path), img)
                    print(f"Saved: {save_path}")
                    index += 1

    except Exception as e:
        # Catch and print any unexpected errors during the process
        print(f"Unexpected error occurred: {e}")

if __name__ == "__main__":
    mobs = get_all_mobs(region=DEFAULT_REGION, version=DEFAULT_VERSION)
    mob_name = input("Enter mob name: ")
    mob_id = find_mob_id(mobs, mob_name)

    mob_file_name = "_".join(text.lower() for text in mob_name.split())

    save_mob(mob_id, folder="monster", mob_name=mob_file_name)