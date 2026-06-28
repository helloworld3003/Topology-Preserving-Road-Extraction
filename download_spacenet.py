import os
from torchgeo.datasets import SpaceNet5

if __name__ == "__main__":
    # Define the folder in the current directory to save data
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spacenet_data")
    os.makedirs(data_dir, exist_ok=True)

    print("Starting download of SpaceNet 5 (Mumbai dataset)...")
    print(f"Data will be saved to: {data_dir}")
    print("This might take a while depending on your internet connection.")
    
    # Since torchgeo's internal download method doesn't use --no-sign-request,
    # and might fail with AWS CLI errors, it's best to download the file manually.
    # Run this command in your terminal first:
    # aws s3 cp s3://spacenet-dataset/spacenet/SN5_roads/tarballs/SN5_roads_train_AOI_8_Mumbai.tar.gz spacenet_data/SN5_roads/train/ --no-sign-request
    
    dataset = SpaceNet5(
        root=data_dir, 
        split="train", 
        aois=[8],           # AOI 8 is Mumbai
        download=False,     # We will download it manually instead
        checksum=False      # Skip checksum since we are handling it manually for now
    )
    
    print("\nDataset successfully initialized from local files!")
    print(f"Total number of samples: {len(dataset)}")
