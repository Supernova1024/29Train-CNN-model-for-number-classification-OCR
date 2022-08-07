# Training Custom CNN model for number classification

This repository includes training code for this project https://github.com/Supernova1024/OCR-PDF-to-CSV-Table-by-CNN-

However, it can be used individually on different projects for number classification.

# Installing
- Download this repository
- Install requirements.txt in project root directory

# Getting Started
- Preparing Datasets
  Copy images of PDF into pdf_img folder.
  Note: To get the images from PDF, please use the pdf_to_img.py script on 

  Then, run below command to get numbers
  ```
  python prepare_data.py
  ```
  Extracted number images are stored in "numbers" folder
  Please copy the number images into sub folders of dataset_32 folder

- Training
  ```
  python train.py
  ``` 
  The trained model is stored in "models" folder

Please give me star if this project was helpful to your startup project. :)



  
