# Building Debris Image Analyzer

This project is designed to scrape images of building debris from Wikimedia Commons, analyze them using machine learning models, and generate a labeled dataset in Excel format.

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

See `requirements.yml` for a list of required Python packages.

## Installation

1. Clone this repository:
git clone https://github.com/yourusername/building-debris-analyzer.git
cd building-debris-analyzer

2. Create a conda environment and install dependencies:

## Usage

1. Activate the conda environment:
conda env create -f requirements.yml
conda activate building-debris-analyzer


2. Run the script:
python wiki_images.py



3. The script will:
- Download images to a folder named `building_debris_images`
- Analyze each image
- Generate an Excel file named `image_labels.xlsx` with the results

## Customization

- To change the search term or number of images, modify the following lines in `wiki_images.py`:
```python
search_term = "building debris damage"
scrape_wikimedia_commons(search_term, num_images=100)


## License

[MIT License](LICENSE)

