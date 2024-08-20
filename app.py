
import streamlit as st
from PIL import Image, ImageDraw
import pandas as pd
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

# Load the pre-trained model
yolo_model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
yolo_model.eval()

# Define the image transformation
transform = T.Compose([T.ToTensor()])

# Streamlit app
st.title('AI Image Segmentation and Object Analysis')
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Open and display the uploaded image
    input_image = Image.open(uploaded_file).convert("RGB")
    st.image(input_image, caption='Uploaded Image', use_column_width=True)

    if st.button('Analyze Image'):
        # Preprocess the image
        image_tensor = transform(input_image).unsqueeze(0)

        # Perform object detection
        with torch.no_grad():
            yolo_predictions = yolo_model(image_tensor)

        # Draw bounding boxes and labels on the image
        draw = ImageDraw.Draw(input_image)
        data_summary = []
        for i in range(len(yolo_predictions[0]['labels'])):
            bbox = yolo_predictions[0]['boxes'][i].cpu().numpy()
            label_text = str(yolo_predictions[0]['labels'][i].item())
            draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline="red", width=2)
            draw.text((bbox[0], bbox[1]), label_text, fill="red")
            data_summary.append({'Object ID': i, 'Label': label_text})

        # Convert data summary to DataFrame
        summary_df = pd.DataFrame(data_summary)

        # Save and display the output image
        output_image_path = 'output_image.jpg'
        input_image.save(output_image_path)
        st.image(output_image_path, caption='Output Image', use_column_width=True)

        # Display the summary DataFrame
        st.dataframe(summary_df)
