import urllib.request
import os
import sys

def download_resume(url, file_path):
    # Check existing file size
    if os.path.exists(file_path):
        downloaded = os.path.getsize(file_path)
    else:
        downloaded = 0
        
    req = urllib.request.Request(url)
    # This header tells AWS to only send the remaining bytes!
    req.add_header('Range', f'bytes={downloaded}-')
    
    try:
        response = urllib.request.urlopen(req)
        
        # Check if the server accepted our partial download request (HTTP 206)
        is_resume = response.getcode() == 206
        
        # Calculate the actual total size
        total_size = int(response.info().get('Content-Length', 0)) + downloaded
        
        mode = 'ab' if is_resume else 'wb'
        if not is_resume and downloaded > 0:
            print("Warning: Server didn't accept resume request. Starting over...")
            downloaded = 0
        else:
            print(f"[*] Resuming download exactly where it dropped off: {downloaded / (1024**3):.2f} GB!")
            
        with open(file_path, mode) as f:
            while True:
                buffer = response.read(8192 * 4) # 32 KB chunks
                if not buffer:
                    break
                f.write(buffer)
                downloaded += len(buffer)
                
                percent = downloaded * 100 / total_size
                sys.stdout.write(f"\rDownloading: {downloaded / (1024**3):.2f} GB / {total_size / (1024**3):.2f} GB ({percent:.1f}%)")
                sys.stdout.flush()
                
        print("\n\nDownload complete!")
    except Exception as e:
        print(f"\nConnection Error: {e}")
        print("Run the script again to resume!")

if __name__ == "__main__":
    url = "https://spacenet-dataset.s3.amazonaws.com/spacenet/SN5_roads/tarballs/SN5_roads_train_AOI_7_Moscow.tar.gz"
    
    # Ensure the directory exists
    output_dir = os.path.join("spacenet_data", "SN5_roads", "train")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "SN5_roads_train_AOI_7_Moscow.tar.gz")
    
    print("Connecting to AWS...")
    download_resume(url, output_file)
