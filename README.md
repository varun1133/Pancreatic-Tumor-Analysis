
```markdown
# 🧠 Pancreatic Tumor Analysis System


## 🚀 Project Overview

Pancreatic cancer is one of the deadliest cancers due to its late diagnosis and rapid progression. This project provides a user-friendly interface for:
- Uploading medical scan images.
- Automatically preprocessing and analyzing them.
- Identifying tumor regions.
- Predicting if the tumor is benign or malignant.
- Providing visual outputs and accuracy confidence.

## 🛠️ Tech Stack

- **Frontend**: HTML, CSS (Grid-based layout)
- **Backend**: Python (Flask), OpenCV, NumPy, scikit-image
- **Machine Learning**: CNN/UNet/ResNet/Inception Models (`model.h5`)
- **Database**: MySQL (`pancreaticdbdb.sql`)
- **Image Processing**: Grayscale, HSV thresholding, segmentation
- **Model Training**: Keras + TensorFlow (see `ModelBuilder.ipynb`)

## 📁 Project Structure

```

├── app.py                 # Flask backend logic
├── inde.html              # Frontend HTML (CSS Grid layout)
├── supportcode.py         # Image comparison and preprocessing functions
├── model.h5               # Trained model
├── pancreaticdbdb.sql     # MySQL schema with user and plant data
├── ModelBuilder.ipynb     # Jupyter notebook for training models
├── static/                # Processed image storage (Grayscale, Binary, etc.)
├── templates/             # HTML templates for the Flask app
├── uploads/               # Uploaded images

````

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/your-username/pancreatic-tumor-analysis.git
cd pancreatic-tumor-analysis
````

### 2. Set up the Python environment

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not present, install manually:

```bash
pip install flask opencv-python numpy pandas matplotlib scikit-image mysql-connector-python tensorflow
```

### 3. Configure MySQL

* Import the `pancreaticdbdb.sql` file into your MySQL server.
* Make sure your `app.py` credentials (host, user, password) match your MySQL setup.

### 4. Run the Flask app

```bash
python app.py
```

Visit `http://localhost:5000` in your browser.

## 📷 Image Upload Flow

1. User registers and logs in.
2. Uploads medical image.
3. Image is processed through thresholding, filters, segmentation.
4. Classification result is shown with model confidence.

## 💡 Features

* User registration and login
* Image similarity computation using SSIM and MSE
* Tumor segmentation using CV + deep learning
* Visual result presentation
* Modular ML architecture (UNet, ResNet, Inception)
  
## 🔐 Security Note

> ⚠️ This project is for **educational/research** purposes only. It is **not intended for clinical use** without validation and regulatory approval.

## 📃 License

This project is licensed under the MIT License.

---

### 👨‍💻 Contributors

* Varun Gowda H S


