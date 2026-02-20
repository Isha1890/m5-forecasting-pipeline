# 🛠️ Setup Guide — Step by Step

Every command below is run in your **Terminal** (Mac/Linux) or **Command Prompt** (Windows).  
You do not need to know how they work — just copy-paste them in order.

---

## What is the Terminal?

- **Mac**: Press `Cmd + Space`, type `Terminal`, hit Enter
- **Windows**: Press `Win + R`, type `cmd`, hit Enter
- **VS Code**: Menu → Terminal → New Terminal (easiest — do this)

---

## PART A — One-Time Setup (15 min)

### A1. Install Python 3.11
Download from **https://python.org/downloads** → Install  
Verify it worked:
```bash
python --version
# Should print: Python 3.11.x
```

### A2. Clone the project
```bash
git clone https://github.com/YOUR_USERNAME/m5-forecasting-pipeline.git
cd m5-forecasting-pipeline
```
> `cd` means "change directory" — you're now inside the project folder

### A3. Create a virtual environment
```bash
# Mac/Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```
> Your terminal prompt should now show `(venv)` at the start — that means it worked

### A4. Install dependencies
```bash
pip install -r requirements.txt
```
> This takes 3-5 minutes. It installs all the ML libraries.

---

## PART B — Kaggle Data (10 min, free)

### B1. Create Kaggle account
Go to **https://www.kaggle.com** → Sign Up (free, use Google login)

### B2. Get your API key
1. Go to **https://www.kaggle.com/settings**
2. Scroll to the **"API"** section
3. Click **"Create New Token"**
4. A file called `kaggle.json` downloads to your Downloads folder

### B3. Move the API key to the right place
```bash
# Mac/Linux — run these 3 lines one at a time:
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Windows (PowerShell) — run these 2 lines:
mkdir $env:USERPROFILE\.kaggle
move $env:USERPROFILE\Downloads\kaggle.json $env:USERPROFILE\.kaggle\kaggle.json
```
> `mkdir` creates a folder. `mv` moves a file. `chmod 600` makes it private (Mac only).

### B4. Accept competition rules (REQUIRED)
Go to **https://www.kaggle.com/c/m5-forecasting-accuracy**  
Scroll down → click **"I Understand and Accept"**  
(Without this step the download will fail with a permissions error)

### B5. Download the data
```bash
pip install kaggle
kaggle competitions download -c m5-forecasting-accuracy -p data/raw/
```
> Downloads ~260MB. Takes 1-5 min depending on your internet.

### B6. Unzip the data
```bash
# Mac/Linux
cd data/raw
unzip m5-forecasting-accuracy.zip
cd ../..

# Windows
# Right-click the zip file in data/raw/ → "Extract All"
# Then come back to the terminal
```

### B7. Verify it worked
```bash
python -c "
import pandas as pd
df = pd.read_csv('data/raw/sales_train_evaluation.csv')
print(f'✅ Real M5 data loaded: {df.shape[0]:,} items × {df.shape[1]:,} columns')
"
```
Expected: `✅ Real M5 data loaded: 30,490 items × 1,947 columns`

### B8. Switch to real data mode
Open `configs/default.yaml` in any text editor and change:
```yaml
data:
  use_demo: false    # ← change true to false
```

---

## PART C — Run It Locally

```bash
# Make sure you're in the project folder with venv active
# You should see (venv) in your terminal prompt

# Run the full pipeline (takes 5-10 min on real data)
python src/pipeline.py

# Launch the dashboard
streamlit run app/dashboard.py
```
Then open **http://localhost:8501** in your browser. That's your live app.

---

## PART D — Deploy to Hugging Face Spaces (free public URL)

### D1. Create HF account
Go to **https://huggingface.co/join** → Sign up (free)

### D2. Create a Space
1. Go to **https://huggingface.co/new-space**
2. Space name: `m5-forecasting`
3. SDK: **Streamlit** ← must select this
4. Visibility: **Public**
5. Click **Create Space**

### D3. Get your HF token
1. **https://huggingface.co/settings/tokens**
2. "New token" → name: `deploy` → Role: **Write**
3. Copy the token (starts with `hf_...`)

### D4. Upload data to HF (handles large files)
```bash
pip install huggingface_hub

# Run this in your terminal (replace the placeholders)
python - << 'EOF'
from huggingface_hub import HfApi, login

login(token="hf_YOUR_TOKEN_HERE")   # paste your token
api = HfApi()

# Create a private dataset to hold the M5 files
api.create_repo(
    repo_id="YOUR_HF_USERNAME/m5-data",
    repo_type="dataset",
    private=True,
)

# Upload the 3 CSVs
for fname in ["sales_train_evaluation.csv", "calendar.csv", "sell_prices.csv"]:
    print(f"Uploading {fname}...")
    api.upload_file(
        path_or_fileobj=f"data/raw/{fname}",
        path_in_repo=fname,
        repo_id="YOUR_HF_USERNAME/m5-data",
        repo_type="dataset",
    )
print("✅ All files uploaded!")
EOF
```

### D5. Add secrets to your Space
1. Go to your Space → **Settings** tab → **"Variables and secrets"**
2. Add these two secrets:
   - Name: `HF_TOKEN` → Value: your `hf_...` token
   - Name: `HF_DATASET` → Value: `YOUR_HF_USERNAME/m5-data`

### D6. Push code to HF Spaces
```bash
# Add HF as a second remote (you keep GitHub too)
git remote add space https://huggingface.co/spaces/YOUR_HF_USERNAME/m5-forecasting

# Push
git push space main
```

HF will build and deploy automatically. Watch progress at:  
`https://huggingface.co/spaces/YOUR_HF_USERNAME/m5-forecasting`

Build takes ~4 minutes. When the status turns green → you're live! 🎉

---

## Common Errors & Fixes

| Error message | What it means | Fix |
|---|---|---|
| `403 Forbidden` | Didn't accept competition rules | Go to kaggle.com/c/m5-forecasting-accuracy and accept |
| `ModuleNotFoundError: src` | Wrong directory | Run `cd m5-forecasting-pipeline` first |
| `No module named kaggle` | Not installed | Run `pip install kaggle` |
| `(venv)` not showing | Virtual env not active | Run `source venv/bin/activate` (Mac) or `venv\Scripts\activate` (Windows) |
| HF build fails | Check logs tab in your Space | Usually a missing package — open an issue |
| Port 8501 in use | Another Streamlit running | Close it or run `streamlit run app/dashboard.py --server.port 8502` |
