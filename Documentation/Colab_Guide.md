# Running on Google Colab (Free T4 GPU)

Google Colab provides free access to NVIDIA T4 GPUs, which will cut your 15-hour CPU training time down to just a few minutes! Because Google Colab has enterprise-grade internet speeds (often 1-2 GB/s), downloading the datasets directly inside Colab is incredibly fast.

Here is the exact step-by-step process to run your pipeline in Colab:

## Step 1: Set Up the Colab Environment
1. Go to [Google Colab](https://colab.research.google.com/) and click **New Notebook**.
2. At the top menu, click **Runtime** > **Change runtime type**.
3. Under Hardware Accelerator, select **T4 GPU** and click Save.

## Step 2: Upload Your Code & Set Kaggle Token
To download the massive dataset directly inside Colab, you will use the official Kaggle API!
1. Go to Kaggle.com -> Account Settings -> **Create New API Token**.
2. Copy the token string they give you (it looks like `KGAT_...`).
3. In Colab, open the left sidebar (Files icon).
4. Drag and drop `train2.py` and `requirements.txt` into the sidebar.

*(Alternatively, if you pushed your code to GitHub, you can just run: `!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git`)*

## Step 3: Run the Pipeline!
In your Colab Notebook, create new code cells and run the following commands exactly in order. Notice the `!` at the start of each command—this tells Colab to run it as a terminal command rather than Python code!

**Cell 1: Navigate and Install Requirements**
*(Because you cloned from GitHub, all your files are inside a new folder. You must use `%cd` to enter that folder before running pip!)*
```bash
%cd Topology-Preserving-Road-Extraction
!pip install -r requirements.txt
```

**Cell 2: Configure Kaggle API & Download DeepGlobe**
*(This uses your new API token, downloads the dataset at 1 GB/s, and unzips it into the `archive` folder!)*
```bash
# Paste your token string here to authenticate!
%env KAGGLE_API_TOKEN=YOUR_TOKEN_STRING_HERE

# Download the dataset directly from Kaggle's servers!
!kaggle datasets download -d balraj98/deepglobe-road-extraction-dataset

# Unzip it directly into the 'archive' folder that train2.py expects
!unzip -q deepglobe-road-extraction-dataset.zip -d archive/
```

**Cell 3: Train the Model on DeepGlobe!**
*(This will automatically detect the T4 GPU and enable AMP!)*
```bash
!python train2.py
```

## Step 4: Download Your Weights
Once `train2.py` finishes its 50 epochs, you will see a new file appear in the left sidebar called `deepglobe_road_model.pth`. 
Right-click that file and select **Download** to save the fully trained brain back to your local computer! 

You can then hook it up to your inference scripts to see the results.
