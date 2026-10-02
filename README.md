🤖 Technology FAQ Chatbot

A simple technology-focused FAQ chatbot built with Python and Streamlit as part of my CodeAlpha Artificial Intelligence Internship.

The chatbot answers common technology questions by comparing a user's question with a predefined collection of frequently asked questions. It uses TF-IDF vectorization and cosine similarity to identify the question that is most similar to the user's input.

✨ Features

- 💬 Interactive chat interface
- 🤖 Answers common technology-related questions
- 🧠 Uses TF-IDF and cosine similarity for question matching
- 🔤 Text preprocessing with tokenization, stop-word removal, and stemming
- 📚 Covers topics including:
  - Python
  - Git and GitHub
  - Artificial Intelligence
  - Machine Learning
  - Deep Learning
  - Neural Networks
  - NLP
  - Computer Vision
  - Cybersecurity
  - Blockchain
  - Cloud Computing
  - Databases
  - Linux
  - Algorithms
  - HTML, CSS and JavaScript
  - Docker and DevOps
  - Computer hardware
  - And more
- 📋 Displays known topics in the sidebar
- 🔄 Includes a button to clear the conversation
- 🛡️ Provides a fallback response when the chatbot cannot confidently match a question

🧠 How It Works

The chatbot follows several steps to process a user's question.

1. Text Preprocessing

The user's question is converted to lowercase and processed using:

- Tokenization
- Stop-word removal
- Stemming

The same preprocessing is applied to the predefined FAQ questions.

2. TF-IDF Vectorization

The processed FAQ questions are converted into numerical vectors using TF-IDF (Term Frequency–Inverse Document Frequency).

TF-IDF represents how important words are within the collection of questions.

3. Cosine Similarity

When a user asks a question, the chatbot compares its TF-IDF vector with the vectors of all stored FAQ questions using cosine similarity.

The question with the highest similarity score is selected as the potential match.

4. Similarity Threshold

The chatbot uses a similarity threshold of 0.30.

If the highest similarity score is below this threshold, the chatbot does not provide a potentially unrelated answer. Instead, it returns a fallback message asking the user to rephrase the question or ask about a supported topic.

🛠️ Technologies Used

- Python — Programming language
- Streamlit — Interactive web application framework
- NLTK — Text processing and stemming
- Scikit-learn — TF-IDF vectorization and cosine similarity
- Regular Expressions ("re") — Basic text processing

📁 Project Structure

technology-faq-chatbot/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore

🚀 Running the Application Locally

Follow the steps below to run the chatbot on your computer.

1. Clone the repository

Open a terminal or command prompt and run:

git clone YOUR_GITHUB_REPOSITORY_URL

Then enter the project directory:

cd technology-faq-chatbot

Replace "YOUR_GITHUB_REPOSITORY_URL" with the URL of this GitHub repository.

2. Create a virtual environment

Creating a virtual environment keeps the project's dependencies separate from other Python projects.

Windows

python -m venv venv

Activate it:

venv\Scripts\activate

macOS / Linux

python3 -m venv venv

Activate it:

source venv/bin/activate

3. Install the dependencies

Install all required packages using:

pip install -r requirements.txt

4. Run the application

Start the Streamlit application with:

streamlit run app.py

Streamlit will provide a local address, usually:

http://localhost:8501

Open the address in your web browser to use the chatbot.

🌐 Live Demo

The chatbot is deployed using Streamlit Community Cloud.

👉 Live Application: https://yosef-technology-faq-chatbot.streamlit.app/

💡 Example Questions

You can ask questions such as:

What is Python?

What is machine learning?

What is GitHub?

What is a neural network?

What is cybersecurity?

What is an API?

What is cloud computing?

What is Docker?

What is RAM?

What is a CPU?
[02/10/2026 20:48] Yosef: The chatbot can also recognize questions that are phrased somewhat differently because it compares the meaning-related word patterns using TF-IDF and cosine similarity.

🎯 Purpose of the Project

This project was developed as part of my CodeAlpha Artificial Intelligence Internship to gain practical experience in:

- Natural language processing
- Text preprocessing
- Information retrieval
- TF-IDF vectorization
- Similarity measurement
- Building interactive Python applications
- Using machine-learning libraries
- Deploying applications with Streamlit
- Managing projects with GitHub

🔮 Possible Future Improvements

Future versions could include:

- 📚 A larger and more diverse knowledge base
- 🧠 More advanced NLP techniques
- 🔎 Improved question matching
- 💾 Persistent conversation history
- 🎤 Voice input
- 🔊 Text-to-speech responses
- 🌍 Support for additional languages
- 🤖 Integration with a large language model for more flexible answers
- 📊 Analytics showing the most frequently asked questions

👨‍💻 Author

Yosef Alene

Developed as part of the CodeAlpha Artificial Intelligence Internship — October 2026.
