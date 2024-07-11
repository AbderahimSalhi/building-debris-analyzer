# Building Debris Image Analyzer

This project scrapes images of building debris from Wikimedia Commons, analyzes them using machine learning models, and generates a labeled dataset in Excel format.

## Features

- Scrapes images from Wikimedia Commons based on a search term
- Downloads and stores images locally
- Analyzes images using CLIP and YOLOv5 models to:
  - Identify disaster types
  - Detect human presence
- Generates an Excel file with image labels and metadata

## Requirements

- Python 3.7+
- CUDA-capable GPU (optional, for faster processing)
- Anaconda or Miniconda

## Installation

1. Clone the repository:
  ```bash
  git clone https://github.com/yourusername/building-debris-analyzer.git
  cd building-debris-analyzer
2. Create a conda environment and install dependencies:
  ```bash
  conda env create -f requirements.yml
  conda activate building-debris-analyzer
3. run the script
  ```bash
  python wiki_images.py
