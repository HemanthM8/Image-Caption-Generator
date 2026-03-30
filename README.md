# Image Caption Generator

A Python application that extracts images from markdown files and generates captions using BLIP (Bootstrapping Language-Image Pre-training) model.

## Features

- Extract image URLs from markdown files (supports `![](url)` syntax)
- Generate captions for images using Salesforce's BLIP base model
- Two interfaces: **CLI** and **Streamlit web app**
- Model caching for fast subsequent runs
- Supports both file uploads and text input

## Requirements

- Python 3.8+
- Virtual environment (venv)

## Installation


### 1. Create and Activate Virtual Environment
```bash
# Create venv
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install transformers[torch] pillow requests markdown streamlit
```

## Usage

### Option 1: CLI Interface

Run the caption generator on a markdown file:

```bash
venv\Scripts\python.exe caption_generator.py sample_files/README.md
```

**Output:**
```
[1] Image URL: https://example.com/image1.jpg
    Caption: a woman with long black hair and blue eyes

[2] Image URL: https://example.com/image2.jpg
    Caption: a woman sitting on a bench with her hand on her hip
```

### Option 2: Streamlit Web Interface

Start the web app:

```bash
venv\Scripts\python.exe -m streamlit run app.py
```

Then open your browser to: **http://localhost:8501**

**Features:**
- Upload markdown files or paste content directly
- Preview images at readable size (300px)
- View generated captions side-by-side
- Auto-generated captions (up to 50 tokens for completeness)

## Project Structure

```
Project/
├── caption_generator.py    # CLI interface
├── app.py                  # Streamlit web interface
├── README.md               # This file
├── sample_images/          # Sample markdown file with images
│   └── README.md
└── venv/                   # Virtual environment (created during setup)
```

## Model Details

- **Model:** Salesforce/blip-image-captioning-base
- **Size:** ~990MB
- **First Run:** Model downloads automatically and is cached locally
- **Task:** Unconditional image captioning (generates caption without prompts)

## Notes

- First run will download the BLIP model (~990MB) - subsequent runs are much faster
- Model is cached in your system's HuggingFace cache directory
- Supports image URLs from web (automatically downloaded for processing)
- Max caption length: 50 tokens

## Troubleshooting

**Issue:** "Module not found" error
- **Solution:** Ensure venv is activated before running scripts

**Issue:** Slow first run
- **Solution:** Normal - model is downloading (~990MB). Subsequent runs are faster.

**Issue:** Incomplete captions
- **Solution:** This is normal. Captions are limited to 50 tokens for brevity. Modify `max_new_tokens` in the code if needed.

