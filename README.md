
# 🧠 Pancreatic Tumor Analysis System

This project is a web-based AI application designed to assist in the early detection and classification of pancreatic tumors using medical imaging. By integrating deep learning with image processing, it aims to provide a reliable diagnostic support tool for healthcare professionals and researchers.

## 📌 Project Features

- Upload medical scan images (e.g., CT or MRI)
- Perform preprocessing including grayscale conversion, HSV transformation, thresholding, and binary masking
- Use trained deep learning models (CNN, UNet, ResNet, Inception) for tumor classification
- Display similarity metrics using SSIM and MSE
- Provide visual output including processed masks and predictions
- User registration and login functionality
- Store and retrieve user data using MySQL

## 🛠️ Technologies Used

- **Frontend**: HTML, CSS
- **Backend**: Python, Flask
- **Image Processing**: OpenCV, scikit-image, NumPy
- **Machine Learning**: TensorFlow, Keras
- **Database**: MySQL
- **Others**: Matplotlib, Pandas, JSON

## ⚙️ Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/pancreatic-tumor-analysis.git
   cd pancreatic-tumor-analysis
Install Python Dependencies

bash
Copy
Edit
pip install flask opencv-python numpy pandas matplotlib scikit-image mysql-connector-python tensorflow
Set Up the Database

Create a MySQL database named pancreaticdbdb

Import the provided pancreaticdbdb.sql file using phpMyAdmin or MySQL CLI:

sql
Copy
Edit
SOURCE pancreaticdbdb.sql;
Run the Application

bash
Copy
Edit
python app.py
Open your browser and go to http://localhost:5000

🧪 How It Works
The user uploads an image.

The system processes the image through various steps (grayscale, thresholding, HSV masking).

A pre-trained deep learning model (model.h5) evaluates the image and predicts the tumor status.

The system returns the prediction along with confidence scores and visualization.

👨‍💻 Author
VARUN GOWDA H S
varungowda1103@gmail.com




