#pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib PyYAML

import os
import io
import zipfile
import shutil
import yaml # For handling data.yaml files
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def authenticate_google_drive():
    """Authenticates with Google Drive API and returns the service object."""
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            print("--- Initiating Google Drive Authentication ---")
            print("A browser window will open for you to log in.")
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
            print("Authentication successful!")
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('drive', 'v3', credentials=creds)

def list_files_in_folder(service, folder_id, mime_type_filter=None):
    """Lists files and their IDs in a specific Google Drive folder."""
    try:
        query = f"'{folder_id}' in parents and trashed = false"
        if mime_type_filter:
            query += f" and mimeType = '{mime_type_filter}'"

        results = service.files().list(
            q=query,
            fields="nextPageToken, files(id, name, mimeType)").execute()
        items = results.get('files', [])

        if not items:
            print(f'No files found in folder ID: {folder_id} with filter {mime_type_filter}.')
            return []
        print(f'\nFiles found in folder ID: {folder_id}:')
        for item in items:
            print(f'  - {item["name"]} (ID: {item["id"]}, Type: {item["mimeType"]})')
        return items
    except HttpError as error:
        print(f'An error occurred while listing files: {error}')
        return []

def download_file(service, file_id, file_name, download_dir='downloads'):
    """Downloads a file from Google Drive."""
    try:
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        request = service.files().get_media(fileId=file_id)
        file_path = os.path.join(download_dir, file_name)
        fh = io.FileIO(file_path, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        print(f'Starting download of "{file_name}"...')
        while done is False:
            status, done = downloader.next_chunk()
            print(f'  Download progress: {int(status.progress() * 100)}%', end='\r')
        print(f'\nFile "{file_name}" downloaded to "{file_path}".')
        return file_path
    except HttpError as error:
        print(f'\nAn error occurred during download: {error}')
        return None

def unzip_file(zip_file_path, extract_to_dir):
    """Unzips a file to a specified directory."""
    try:
        if not os.path.exists(extract_to_dir):
            os.makedirs(extract_to_dir)

        print(f"Unzipping '{zip_file_path}' to '{extract_to_dir}'...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # Check for the top-level 'dataset' folder to extract its contents directly
            # This handles cases where 'dataset' is the root, or where files are directly in zip
            namelist = zip_ref.namelist()
            if namelist and namelist[0].startswith('dataset/'):
                # It has a top-level 'dataset/' folder. Extract all and then move contents.
                zip_ref.extractall(extract_to_dir)
                source_path = os.path.join(extract_to_dir, 'dataset')
                # Move contents of 'dataset/' to 'extract_to_dir'
                for item in os.listdir(source_path):
                    shutil.move(os.path.join(source_path, item), extract_to_dir)
                shutil.rmtree(source_path) # Remove the empty 'dataset' folder
            else:
                # No top-level 'dataset/' folder, extract directly
                zip_ref.extractall(extract_to_dir)
        print(f"Successfully unzipped '{zip_file_path}'.")
        return True
    except zipfile.BadZipFile:
        print(f"Error: '{zip_file_path}' is not a valid zip file.")
        return False
    except Exception as e:
        print(f"An error occurred while unzipping '{zip_file_path}': {e}")
        return False

def combine_yolo_datasets(extracted_base_dir, combined_dataset_output_dir='combined_yolo_dataset'):
    """
    Combines multiple YOLO-formatted datasets into a single dataset.
    Ensures unique filenames and merges data.yaml.
    """
    print("\n--- Combining YOLO datasets ---")

    # Define paths for the combined dataset
    combined_images_train_dir = os.path.join(combined_dataset_output_dir, 'images', 'train')
    combined_images_val_dir = os.path.join(combined_dataset_output_dir, 'images', 'val')
    combined_labels_train_dir = os.path.join(combined_dataset_output_dir, 'labels', 'train')
    combined_labels_val_dir = os.path.join(combined_dataset_output_dir, 'labels', 'val')
    combined_data_yaml_path = os.path.join(combined_dataset_output_dir, 'data.yaml')

    # Create target directories if they don't exist
    for path in [combined_images_train_dir, combined_images_val_dir,
                 combined_labels_train_dir, combined_labels_val_dir]:
        os.makedirs(path, exist_ok=True)

    all_class_names = set()
    num_merged_images = {'train': 0, 'val': 0}

    # Iterate through each extracted dataset
    for i, dataset_dir_name in enumerate(os.listdir(extracted_base_dir)):
        dataset_path = os.path.join(extracted_base_dir, dataset_dir_name)
        if not os.path.isdir(dataset_path):
            continue

        print(f"\nProcessing dataset from: {dataset_path}")

        # 1. Merge Images and Labels
        for split in ['train', 'val']:
            src_images_dir = os.path.join(dataset_path, 'images', split)
            src_labels_dir = os.path.join(dataset_path, 'labels', split)
            dest_images_dir = os.path.join(combined_dataset_output_dir, 'images', split)
            dest_labels_dir = os.path.join(combined_dataset_output_dir, 'labels', split)

            if os.path.exists(src_images_dir) and os.path.exists(src_labels_dir):
                for img_filename in os.listdir(src_images_dir):
                    name_without_ext, img_ext = os.path.splitext(img_filename)
                    label_filename = name_without_ext + '.txt'

                    src_img_path = os.path.join(src_images_dir, img_filename)
                    src_label_path = os.path.join(src_labels_dir, label_filename)

                    if os.path.isfile(src_img_path) and os.path.isfile(src_label_path):
                        # Create unique filenames for the combined dataset
                        new_img_filename = f"ds{i+1}_{img_filename}"
                        new_label_filename = f"ds{i+1}_{label_filename}"

                        dest_img_path = os.path.join(dest_images_dir, new_img_filename)
                        dest_label_path = os.path.join(dest_labels_dir, new_label_filename)

                        shutil.copy2(src_img_path, dest_img_path)
                        shutil.copy2(src_label_path, dest_label_path)
                        num_merged_images[split] += 1
                        # print(f"  Copied {img_filename} and {label_filename} to {split} set.")
            else:
                print(f"  Warning: No '{split}' images/labels found in {dataset_path}")

        # 2. Extract Class Names from data.yaml
        data_yaml_path = os.path.join(dataset_path, 'data.yaml')
        if os.path.exists(data_yaml_path):
            try:
                with open(data_yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                    if 'names' in data and isinstance(data['names'], list):
                        all_class_names.update(data['names'])
                        print(f"  Extracted class names from {data_yaml_path}: {data['names']}")
            except Exception as e:
                print(f"  Error reading {data_yaml_path}: {e}")
        else:
            print(f"  Warning: data.yaml not found in {dataset_path}")

    # 3. Create the combined data.yaml
    combined_data_yaml_content = {
        'path': os.path.abspath(combined_dataset_output_dir), # Absolute path to the combined dataset
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(all_class_names),
        'names': sorted(list(all_class_names)) # Sort for consistent order
    }

    with open(combined_data_yaml_path, 'w') as f:
        yaml.dump(combined_data_yaml_content, f, sort_keys=False) # Keep order for 'train', 'val', 'nc', 'names'
    print(f"\nCreated combined data.yaml at: {combined_data_yaml_path}")
    print(f"  Total unique classes: {len(all_class_names)}")
    print(f"  Total train images merged: {num_merged_images['train']}")
    print(f"  Total val images merged: {num_merged_images['val']}")

    print(f"\nCombined dataset created successfully in: {combined_dataset_output_dir}")
    print("You can now point your YOLO model to this directory and the data.yaml file.")

def main():
    """Main function to orchestrate the download, unzip, and combine process."""
    service = authenticate_google_drive()

    # --- IMPORTANT: Replace with your actual Google Drive folder ID ---
    google_drive_folder_id = '1bUkIYQRXX08OKI5TuOSg-eqntSudGaFB' # <--- REPLACE THIS!

    if google_drive_folder_id == '1bUkIYQRXX08OKI5TuOSg-eqntSudGaFB':
        print("\n!!! IMPORTANT: Please update 'google_drive_folder_id' in the script with your actual Google Drive folder ID. !!!")
        print("The script cannot proceed without a valid folder ID.")
        return

    # Define directories for downloads and extractions
    downloads_dir = 'downloaded_zips'
    extracted_base_dir = 'extracted_yolo_data'
    combined_output_dir = 'combined_yolo_dataset'

    # Clean up previous runs if they exist, for a fresh start
    if os.path.exists(downloads_dir):
        shutil.rmtree(downloads_dir)
    if os.path.exists(extracted_base_dir):
        shutil.rmtree(extracted_base_dir)
    if os.path.exists(combined_output_dir):
        shutil.rmtree(combined_output_dir)

    os.makedirs(downloads_dir, exist_ok=True)
    os.makedirs(extracted_base_dir, exist_ok=True)

    # 1. List and Download Zip Files
    print(f"\n--- Searching for ZIP files in Google Drive folder (ID: {google_drive_folder_id}) ---")
    zip_mime_type = 'application/zip'
    zip_files_info = list_files_in_folder(service, google_drive_folder_id, mime_type_filter=zip_mime_type)

    downloaded_zip_paths = []
    if zip_files_info:
        for file_info in zip_files_info:
            if file_info['mimeType'] == zip_mime_type:
                print(f"\n--- Found ZIP file: {file_info['name']} ---")
                download_path = download_file(service, file_info['id'], file_info['name'], download_dir=downloads_dir)
                if download_path:
                    downloaded_zip_paths.append(download_path)
    else:
        print("No ZIP files found in the specified folder.")
        return

    if len(downloaded_zip_paths) < 1: # You mentioned 3, but this will work with any number
        print("Not enough zip files downloaded to combine. Need at least one.")
        return

    # 2. Unzip Files
    for i, zip_path in enumerate(downloaded_zip_paths):
        # Create a unique directory for each zip's contents
        # This will be e.g., extracted_yolo_data/zip_content_1, /zip_content_2, etc.
        extract_target_dir = os.path.join(extracted_base_dir, f'zip_content_{i+1}')
        unzip_file(zip_path, extract_target_dir)

    # 3. Combine Datasets
    combine_yolo_datasets(extracted_base_dir, combined_output_dir)

    # 4. Optional: Clean up intermediate files
    print(f"\n--- Cleaning up intermediate files ---")
    if os.path.exists(downloads_dir):
        shutil.rmtree(downloads_dir)
        print(f"Removed '{downloads_dir}'")
    if os.path.exists(extracted_base_dir):
        shutil.rmtree(extracted_base_dir)
        print(f"Removed '{extracted_base_dir}'")
    print("Cleanup complete. Your combined dataset is in the 'combined_yolo_dataset' folder.")

if __name__ == '__main__':
    main()