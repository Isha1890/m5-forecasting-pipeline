# 🚀 Complete GitHub Deployment Guide
## M5 Time-Series Forecasting Pipeline

This guide walks you through getting the full project live on GitHub — from zero to a polished, public portfolio repository.

---

## STEP 1 — Create the GitHub Repository

1. Go to **https://github.com/new**
2. Fill in:
   - **Repository name**: `m5-forecasting-pipeline`
   - **Description**: `Production-grade demand forecasting pipeline on M5 dataset — ARIMA, Prophet, LightGBM, LSTM + Streamlit dashboard`
   - **Visibility**: Public ✅
   - **Do NOT** check "Add README" (we have our own)
3. Click **Create repository**
4. Copy your repo URL: `https://github.com/YOUR_USERNAME/m5-forecasting-pipeline.git`

---

## STEP 2 — Set Up Local Git

Open terminal in the project root (`m5-forecasting/`):

```bash
# Initialize git
git init

# Set your identity (if not already set globally)
git config user.name "Your Name"
git config user.email "your@email.com"

# Add remote
git remote add origin https://github.com/YOUR_USERNAME/m5-forecasting-pipeline.git

# Stage all files
git add .

# First commit
git commit -m "feat: initial commit — M5 forecasting pipeline with ARIMA, Prophet, LightGBM, LSTM + Streamlit"

# Push to main
git branch -M main
git push -u origin main
```

---

## STEP 3 — Update README with Your GitHub Username

Open `README.md` and replace `YOUR_USERNAME` in the clone URL:

```bash
# Quick replace (macOS/Linux)
sed -i '' 's/YOUR_USERNAME/your_actual_username/g' README.md

# Then commit
git add README.md
git commit -m "docs: update GitHub username in README"
git push
```

---

## STEP 4 — Add Repository Topics (Makes You Discoverable)

1. Go to your repo on GitHub
2. Click the **⚙️ gear icon** next to "About" on the right sidebar
3. Add these topics:
   ```
   machine-learning  time-series  forecasting  python  lightgbm
   lstm  prophet  streamlit  m5-competition  demand-forecasting
   data-science  pytorch  arima  pandas  portfolio
   ```
4. Add website URL if you deploy to Streamlit Cloud (see Step 6)

---

## STEP 5 — Pin to Your Profile

1. Go to **your GitHub profile** (`github.com/YOUR_USERNAME`)
2. Click **"Customize your pins"**
3. Select `m5-forecasting-pipeline`
4. It now appears front-and-center on your profile

---

## STEP 6 — Deploy Streamlit App (Free, Public URL)

This gives you a live demo link to put on your resume.

### Option A: Streamlit Community Cloud (Recommended — 100% Free)

1. Go to **https://share.streamlit.io**
2. Sign in with GitHub
3. Click **"New app"**
4. Configure:
   - **Repository**: `YOUR_USERNAME/m5-forecasting-pipeline`
   - **Branch**: `main`
   - **Main file path**: `app/dashboard.py`
5. Click **Deploy**

The app will be live at:
```
https://your-username-m5-forecasting-pipeline-app-dashboard-XXXXX.streamlit.app
```

6. Add this URL to:
   - Your GitHub repo's "About" section (website field)
   - Your README badge
   - Your LinkedIn and resume

### Important for Streamlit Cloud
The demo data generator runs automatically. Add this `packages.txt` to your repo root:

```bash
echo "" > packages.txt  # No system packages needed
```

And create `.streamlit/config.toml`:
```toml
[theme]
base = "dark"
primaryColor = "#4fc3f7"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#1e2130"
textColor = "#e0e0e0"
```

```bash
mkdir -p .streamlit
cat > .streamlit/config.toml << 'EOF'
[theme]
base = "dark"
primaryColor = "#4fc3f7"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#1e2130"
textColor = "#e0e0e0"
EOF

git add .streamlit/
git commit -m "feat: add Streamlit theme config"
git push
```

---

## STEP 7 — Add a Demo GIF to README (High Impact)

Recruiters decide in 5 seconds. A GIF of your dashboard running is instant credibility.

1. Run your Streamlit app locally
2. Record a 10-15 second GIF using:
   - **macOS**: [Kap](https://getkap.co/) (free)
   - **Windows**: [ScreenToGif](https://www.screentogif.com/) (free)
   - **All platforms**: [Loom](https://loom.com) → export GIF
3. Save as `assets/demo.gif`
4. Add to your README at the top:

```markdown
![Dashboard Demo](assets/demo.gif)
```

```bash
mkdir -p assets
# Copy your demo.gif into assets/
git add assets/demo.gif
git commit -m "docs: add dashboard demo GIF"
git push
```

---

## STEP 8 — Add a License

```bash
# MIT License is standard for portfolio projects
curl https://raw.githubusercontent.com/nicholasjclark/nicholasjclark.github.io/master/LICENSE -o LICENSE

# Edit to add your name and year
# Then:
git add LICENSE
git commit -m "chore: add MIT license"
git push
```

---

## STEP 9 — Workflow for Future Changes

```bash
# Always work on feature branches
git checkout -b feat/add-xgboost-model

# Make changes, then:
git add .
git commit -m "feat: add XGBoost model with hyperparameter tuning"
git push origin feat/add-xgboost-model

# Create Pull Request on GitHub → merge to main
# This shows professional Git workflow to reviewers
```

---

## STEP 10 — Resume & LinkedIn Bullet Points

Copy these directly:

**Resume (Projects section):**
```
M5 Demand Forecasting Pipeline                               Nov 2025 | github.com/USERNAME/m5-forecasting-pipeline
• Built end-to-end forecasting pipeline on M5 (Walmart) dataset with 50K+ daily sales records across
  10 stores; implemented walk-forward backtesting achieving 9.3% MAPE with ensemble model
• Engineered 40+ features (lag, rolling statistics, cyclical calendar, target encoding, price promotions)
  to capture weekly/annual seasonality and promotional demand spikes
• Trained and benchmarked 4 models (ARIMA, Prophet, LightGBM, LSTM/PyTorch); LightGBM reduced
  MAE by 49% vs. ARIMA baseline; built Streamlit dashboard with interactive forecast visualization
• Implemented modular, production-style codebase with typed functions, unit tests (pytest), and CI/CD via
  GitHub Actions; deployed live demo on Streamlit Community Cloud
```

**LinkedIn Featured section:** Add the Streamlit app URL as a "Link" with a screenshot.

---

## 🎯 Final Checklist

- [ ] Repo created and code pushed
- [ ] Topics added (discoverable in GitHub search)
- [ ] Pinned on your profile
- [ ] Streamlit app deployed (live URL)
- [ ] Demo GIF in README
- [ ] LinkedIn Featured updated
- [ ] Resume updated

**You're done. This project alone can get you interviews at FAANG.**
