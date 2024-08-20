# hritika-vaishnav-wasserstoff-AiInternTask

## AI Pipeline for Image Segmentation and Object Analysis
 
### Project Overview 
This project involves building a comprehensive AI pipeline for image segmentation and object analysis. The pipeline uses state-of-the-art deep learning models to: 
1. Segment objects within an input image. 
2. Extract and store each segmented object.
3. Identify and describe each object.
4. Extract text/data from the objects. 
5. Summarize object attributes. 
6. Map and visualize the extracted data. 

*The pipeline is implemented using popular machine learning and computer vision libraries, and it includes a Streamlit UI for interactive testing.*

### Setup Instructions 
•	Prerequisites - Python 3.7 or higher - Git (optional, for cloning the repository)
•	Clone the Repository ```bash git clone https://github.com/yourusername/ai-image-pipeline.git cd ai-image-pipeline

### Install Dependencies
Ensure you have pip installed. Then, install the required Python packages:
pip install -r requirements.txt

### Usage Guidelines
#### Running the Pipeline
1.	Start Streamlit App
streamlit run streamlit_app/app.py
1.	This will open the Streamlit UI in your web browser.
2.	Upload Image
o	Use the upload button to select an input image for processing.
3.	View Results
o	The segmented objects will be displayed on the original image.
o	You can review extracted object images, descriptions, and text/data.
o	The final output image with annotations and a summary table will be generated.

#### Project Structure
project_root/
│
├── data/
│   ├── input_images/               # Directory for input images
│   ├── segmented_objects/          # Directory to save segmented object images
│   └── output/                     # Directory for output images and tables
│
├── models/
│   ├── segmentation_model.py       # Script for segmentation model
│   ├── identification_model.py     # Script for object identification model
│   ├── text_extraction_model.py    # Script for text/data extraction model
│   └── summarization_model.py      # Script for summarization model
│
├── utils/
│   ├── preprocessing.py            # Script for preprocessing functions
│   ├── postprocessing.py           # Script for postprocessing functions
│   ├── data_mapping.py             # Script for data mapping functions
│   └── visualization.py            # Script for visualization functions
│
├── streamlit_app/
│   ├── app.py                      # Main Streamlit application script
│   └── components/                 # Directory for Streamlit components
│
├── tests/
│   ├── test_segmentation.py        # Tests for segmentation
│   ├── test_identification.py      # Tests for identification
│   ├── test_text_extraction.py     # Tests for text extraction
│   └── test_summarization.py       # Tests for summarization
│
├── README.md                       # Project overview and setup instructions
├── requirements.txt                # Required Python packages
└── presentation.pptx               # Presentation slides summarizing the project

#### Dependencies
•	torchvision
•	matplotlib
•	opencv-python
•	PIL
•	pandas
•	streamlit
•	scikit-learn
•	Tesseract OCR or EasyOCR
See requirements.txt for the complete list of dependencies.

#### Testing
To run tests for the pipeline, use
pytest tests/

