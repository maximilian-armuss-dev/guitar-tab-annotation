# 🎸 Guitar Tab Analyzer
🚧 Work in Progress – Extract and analyze guitar tab images using computer vision and deep learning.


## 🎯 Project Goals

There are countless ways to play the same chord on a guitar — but not all are practical in context. 
Jumping from a chord on the 1st fret to one on the 11th? Not ideal.

This project automates the annotation of guitar tabs by extracting finger positions, fret positions, and fret offsets. 
These annotations will support a separate project that suggests the most playable chord variations for any given sequence — making transitions smoother and more intuitive.

## 📌 Overview
This project automates the extraction and annotation of guitar chord tabs from PDF files. It consists of two main steps:

1. 📄 **Screenshot Extraction**
   
   Automatically grab play instruction images from a PDF.


2. 🧠 **CNN-Based Annotation**

   Predict the following features from screenshots using trained models:
      + 🖐️ Finger Position
      + 🎸 Fret Position
      + ↕️ Fret Offset

Each annotation task is handled in a separate notebook for modularity and clarity.

## 📁 Project Structure

    tab-annotation-model/
    │
    ├── data/
    │   ├── dataset/
    │   │   ├── frets/
    │   │   └── tabs/
    │   ├── Guitar Chords Galore.pdf
    │   ├── labelled_fret_offsets.csv
    │   └── labelled_tabs.csv
    │
    ├── src/
    │   ├── notebooks/
    │   │   ├── screenshots.ipynb                   # Extract play instruction images from the PDF.
    │   │   ├── testing.ipynb                       # Run model inference on new screenshots
    │   │   ├── train_finger_pos_model.ipynb        # Train CNN to predict finger placements.
    │   │   ├── train_fret_offset_model.ipynb       # Train CNN to estimate fretboard offset.
    │   │   └── train_fret_pos_model.ipynb          # Train CNN to detect fret positions.
    │   ├── create_screenshots.py                   # Script to extract screenshots from PDF
    │   ├── datasets.py                             # Dataset loaders and preprocessing
    │   ├── model.py                                # CNN model definitions
    │   └── util.py                                 # Helper functions
    │
    ├── .gitignore
    ├── environment.yml                             # Conda environment setup
    └── README.md

## 📦 Setup

Create the environment:

``
conda env create -f environment.yml
``

``
conda activate tab-annotation
``

## 🔄 Workflow

Extract screenshots from Guitar Chords Galore.pdf using `screenshots.ipynb` or `create_screenshots.py`

Train models using the labeled data in `labelled_tabs.csv` and `labelled_fret_offsets.csv`

Test and evaluate using `testing.ipynb`