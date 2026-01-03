# Liver Damage Detection using Deep Learning

This project aims to automatically detect liver damage from medical images using advanced deep learning techniques. It classifies liver conditions into three categories: **Normal Liver**, **Hepatocellular Carcinoma (HCC)**, and **Cholangiocarcinoma (CC)**.

## Project Structure

The project is organized as follows:

- **`code/`**: Contains the source code.
  - `app.py`: A Streamlit web application for real-time inference.
  - `Code.ipynb`: Jupyter notebook used for training/analysis.
- **`data/`**: dataset images categorized by class (`Normal`, `HCC`, `CC`).
- **`model/`**: Contains the trained deep learning model (`VGG19_final_model.keras`).
- **`requirements.txt`**: List of Python dependencies.

## Installation

1.  **Clone or download** this repository.
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Web App

To launch the interactive web interface for liver damage detection:

```bash
streamlit run code/app.py
```

This will open a browser window where you can upload liver images and get real-time predictions.

## Model Details

- **Architecture**: VGG19 (Transfer Learning)
- **Input**: Histopathology images (224x224)
- **Classes**: Normal, CC, HCC

## Acknowledgements

This project was built for research and academic purposes as part of a study in medical image analysis.
