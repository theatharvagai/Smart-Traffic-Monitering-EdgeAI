import os
import urllib.request

def download_model(url, filename):
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filename)
    print(f"Successfully downloaded {filename}")

if __name__ == "__main__":
    # Example: You can replace these with your own direct download links
    # For now, I'll leave them as placeholders.
    models = {
        "personal_model.pt": "YOUR_DIRECT_DOWNLOAD_LINK_HERE",
        "personal_model.onnx": "YOUR_DIRECT_DOWNLOAD_LINK_HERE"
    }
    
    for filename, url in models.items():
        if not os.path.exists(filename):
            if "YOUR_DIRECT" not in url:
                download_model(url, filename)
            else:
                print(f"Skipping {filename}: Please update the download link in fetch_model.py")
        else:
            print(f"{filename} already exists.")
