Phishing Email Detection
An interactive application that detects phishing emails based on text input using a machine learning model trained on a Kaggle dataset.

Features
Identifies phishing vs. legitimate (ham) emails.
Interactive web app built with Streamlit.
Real-time email classification based on text input.

Project Structure
.
├── app.py                # Main script to run the Streamlit app
├── requirements.txt      # Python dependencies
├── model.pkl             # Trained machine learning model
├── vectorizer.pkl        # TF-IDF vectorizer
├── README.md             # Project documentation

How to Set Up the Project
Prerequisites
Ensure you have the following installed:

Python 3.8 or later
pip (Python package installer)

Clone the Repository
git clone https://github.com/your-username/phishing-email-detection.git
cd phishing-email-detection

Install Dependencies
Install required libraries using pip:
pip install -r requirements.txt


Download NLTK Data
Run this command to download required NLTK resources:
import nltk
nltk.download('punkt')
nltk.download('stopwords')

Run the Application
Start the Streamlit app:
streamlit run app.py

The application will launch in your default web browser at http://localhost:8501.

Dataset
The model is trained using a Kaggle dataset that contains labeled email data.

Source: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset


Technologies Used
Programming Language: Python
Machine Learning Library: Scikit-learn
Data Processing: Pandas, NLTK
Vectorization: TF-IDF (Term Frequency-Inverse Document Frequency)
Web Framework: Streamlit
Deployment: Streamlit Cloud


How It Works
Input Preprocessing: The text input is tokenized, cleaned, and vectorized using TF-IDF.
Prediction: The trained machine learning model (Multinomial Naive Bayes) classifies the input as either phishing or ham.
Output: Displays whether the email is phishing or legitimate.


Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a feature branch (git checkout -b feature-name).
Commit changes (git commit -m "Add feature").
Push to the branch (git push origin feature-name).
Open a pull request.


