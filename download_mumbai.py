import urllib.request
import os
import sys
import tarfile

def download_resume(url, file_path):
    if os.path.exists(file_path):
        downloaded = os.path.getsize(file_path)
    else:
        downloaded = 0
        
    req = urllib.request.Request(url)
    # AWS S3 supports Range requests, so if the connection drops, we resume right where we left off!
    req.add_header('Range', f'bytes={downloaded}-')
    
    try:
        response = urllib.request.urlopen(req)
        
        is_resume = response.getcode() == 206
        
        # Calculate actual total size
        total_size = int(response.info().get('Content-Length', 0)) + downloaded
        
        # If the file is already fully downloaded
        if downloaded == total_size and total_size > 0:
            print("[+] Mumbai dataset is already fully downloaded!")
            return True
            
        mode = 'ab' if is_resume else 'wb'
        if not is_resume and downloaded > 0:
            print("[!] Server didn't accept resume request. Starting over...")
            downloaded = 0
        elif downloaded > 0:
            print(f"[*] Resuming download exactly where it dropped off: {downloaded / (1024**3):.2f} GB!")
            
        with open(file_path, mode) as f:
            while True:
                buffer = response.read(8192 * 8) # 64 KB chunks for faster download
                if not buffer:
                    break
                f.write(buffer)
                downloaded += len(buffer)
                
                percent = downloaded * 100 / total_size
                sys.stdout.write(f"\rDownloading Mumbai: {downloaded / (1024**3):.2f} GB / {total_size / (1024**3):.2f} GB ({percent:.1f}%)")
                sys.stdout.flush()
                
        print("\n\n[+] Download complete!")
        return True
    except Exception as e:
        print(f"\n[!] Connection Error: {e}")
        print("Just run the script again and it will resume from where it failed!")
        return False

def extract_tarball(file_path, extract_path):
    print(f"\n[*] Extracting {file_path}...")
    print("    This may take a few minutes for a multi-gigabyte dataset...")
    try:
        with tarfile.open(file_path, "r:gz") as tar:
            # We extract it directly into the spacenet_data folder so torchgeo finds it!
            tar.extractall(path=extract_path)
        print("[+] Extraction completely finished!")
        
        # Optional: Delete the heavy tarball to save hard drive space
        print("[*] Cleaning up the heavy .tar.gz file to save space...")
        os.remove(file_path)
        print("[+] Cleanup complete. You are ready to train!")
        
    except Exception as e:
        print(f"[!] Extraction Error: {e}")

if __name__ == "__main__":
    # SpaceNet 5 Public AWS S3 URL for Mumbai
    url = "https://spacenet-dataset.s3.amazonaws.com/spacenet/SN5_roads/tarballs/SN5_roads_train_AOI_8_Mumbai.tar.gz"
    
    # torchgeo expects SpaceNet5 data to be placed like this:
    # spacenet_data/SN5_roads/train/SN5_roads_train_AOI_8_Mumbai.tar.gz (or extracted folder)
    output_dir = os.path.join("spacenet_data") 
    tarball_dir = os.path.join(output_dir, "SN5_roads", "train")
    os.makedirs(tarball_dir, exist_ok=True)
    
    tarball_path = os.path.join(tarball_dir, "SN5_roads_train_AOI_8_Mumbai.tar.gz")
    
    print("\n========== SPACENET 5 MUMBAI DOWNLOADER ==========")
    print("This script natively connects to AWS and downloads the heavy dataset.")
    print("It features Auto-Resume, so if your internet drops, just run it again!\n")
    
    success = download_resume(url, tarball_path)
    
    # If download was fully successful (or already existed), extract it automatically!
    if success:
        # Check if it was already extracted (meaning a folder exists instead of a tarball)
        extracted_folder = os.path.join(tarball_dir, "AOI_8_Mumbai")
        if os.path.exists(extracted_folder) and not os.path.exists(tarball_path):
            print("\n[+] Mumbai data is already extracted and ready to go!")
        else:
            extract_tarball(tarball_path, tarball_dir)
