import re
import streamlit as st
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------------
# 1) FAQ DATA  (question, answer)  -> add your own at the end of the list
# ------------------------------------------------------------------
FAQS = [
    ("What is GitHub?",
     "GitHub is a website where developers store, share and collaborate on code. It is built on Git and lets teams track changes, review code and work on projects together."),
    ("What is Git?",
     "Git is a free version control system that records every change made to your code, so you can go back to older versions and work with others without overwriting each other's work."),
    ("What is Python?",
     "Python is a popular, beginner-friendly programming language known for its simple, readable syntax. It is widely used in AI, data science, web development and automation."),
    ("What is artificial intelligence (AI)?",
     "Artificial intelligence is the ability of computers to perform tasks that normally need human intelligence, such as understanding language, recognizing images, making decisions and learning from data."),
    ("What is machine learning?",
     "Machine learning is a branch of AI where computers learn patterns from data and improve at a task without being explicitly programmed for every rule."),
    ("What is deep learning?",
     "Deep learning is a type of machine learning that uses neural networks with many layers. It powers speech recognition, image recognition and modern chatbots."),
    ("What is a neural network?",
     "A neural network is a computing model inspired by the human brain. It is made of connected layers of nodes (neurons) that learn to recognize patterns in data."),
    ("What is natural language processing (NLP)?",
     "NLP is the field of AI that helps computers understand, interpret and generate human language. Translation, chatbots and spell checkers all use it."),
    ("What is computer vision?",
     "Computer vision is the field of AI that enables computers to understand images and videos, for example detecting objects, recognizing faces or reading text."),
    ("What is generative AI?",
     "Generative AI creates new content such as text, images, music or code, based on patterns learned from large amounts of training data. ChatGPT and Claude are examples."),
    ("What is a chatbot?",
     "A chatbot is a program that talks with users through text or voice. Simple ones match questions to ready answers, while advanced ones use AI to generate replies."),
    ("What is data science?",
     "Data science combines statistics, programming and domain knowledge to collect, clean, analyze and visualize data in order to find useful insights."),
    ("What is big data?",
     "Big data refers to extremely large and complex datasets that are too big for traditional tools. It is usually described by volume, velocity and variety."),
    ("What is an API (application programming interface)?",
     "An API is a set of rules that lets two software programs talk to each other. For example, a weather app uses an API to get data from a weather service."),
    ("What is cloud computing?",
     "Cloud computing means using servers, storage and software over the internet instead of your own computer. Examples include Google Drive, AWS and Microsoft Azure."),
    ("What is cybersecurity?",
     "Cybersecurity is the practice of protecting computers, networks and data from attacks, theft and damage."),
    ("What is a firewall?",
     "A firewall is a security system that monitors and controls network traffic, blocking suspicious connections based on a set of rules."),
    ("What is a VPN (virtual private network)?",
     "A VPN creates an encrypted connection between your device and the internet, hiding your IP address and protecting your data on public networks."),
    ("What is encryption?",
     "Encryption scrambles data into unreadable code so that only someone with the correct key can read it. It protects messages, passwords and payments."),
    ("What is phishing?",
     "Phishing is a scam where attackers send fake emails or messages that look genuine to trick you into revealing passwords or personal information."),
    ("What is blockchain?",
     "A blockchain is a shared digital ledger that records transactions in linked blocks. Once recorded, data is very hard to change. Cryptocurrencies like Bitcoin use it."),
    ("What is the Internet of Things (IoT)?",
     "IoT is a network of everyday physical devices, such as smart watches, thermostats and cameras, that connect to the internet and exchange data."),
    ("What is 5G?",
     "5G is the fifth generation of mobile networks. It offers much faster speeds, lower delay and the ability to connect many more devices than 4G."),
    ("What is a database?",
     "A database is an organized collection of data stored electronically so it can be easily searched, updated and managed. Examples are MySQL and MongoDB."),
    ("What is SQL?",
     "SQL (Structured Query Language) is the standard language used to create, read, update and delete data in relational databases."),
    ("What is an operating system?",
     "An operating system is the main software that manages a computer's hardware and runs applications. Windows, macOS, Linux, Android and iOS are examples."),
    ("What is Linux?",
     "Linux is a free, open-source operating system kernel used in servers, smartphones (Android), and many developer machines. It is known for stability and security."),
    ("What is an algorithm?",
     "An algorithm is a step-by-step set of instructions for solving a problem or completing a task, like a recipe for a computer."),
    ("What is open source software?",
     "Open source software has its code publicly available so anyone can view, use, modify and share it. Linux and Python are open source."),
    ("What is HTML?",
     "HTML (HyperText Markup Language) is the standard language used to create the structure and content of web pages."),
    ("What is CSS?",
     "CSS (Cascading Style Sheets) controls how web pages look, including colors, fonts and layout."),
    ("What is JavaScript?",
     "JavaScript is a programming language that makes websites interactive, such as menus, animations and forms. It runs in the browser and on servers."),
    ("What is Docker?",
     "Docker is a tool that packages an application with everything it needs into a container, so it runs the same way on any computer."),
    ("What is DevOps?",
     "DevOps is a way of working that combines software development and IT operations to build, test and release software faster and more reliably."),
    ("What is an IP address?",
     "An IP address is a unique number that identifies a device on a network or the internet, so data knows where to be sent."),
    ("What is RAM?",
     "RAM (Random Access Memory) is a computer's short-term memory. It temporarily holds data of running programs for fast access and is cleared when the computer is turned off."),
    ("What is a CPU?",
     "The CPU (Central Processing Unit) is the main chip of a computer that carries out instructions and calculations. It is often called the brain of the computer."),
    ("What is Streamlit?",
     "Streamlit is a free Python library that lets you build interactive web apps for data and AI projects using only Python code."),
]

FALLBACK = ("Sorry, I couldn't find an answer to that. "
            "Try rephrasing, or ask about a topic from the list in the sidebar.")
THRESHOLD = 0.30  # minimum similarity (0 to 1) needed to answer
# ------------------------------------------------------------------
# 2) PREPROCESSING  (lowercase, tokenize, remove stop words, stem)
# ------------------------------------------------------------------
stemmer = PorterStemmer()
STOP = set(ENGLISH_STOP_WORDS) | {"tell", "explain", "define", "meaning", "mean", "means"}


def preprocess(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return " ".join(stemmer.stem(w) for w in words if w not in STOP)


# ------------------------------------------------------------------
# 3) MATCHING  (TF-IDF vectors + cosine similarity)
# ------------------------------------------------------------------
@st.cache_resource
def build_model():
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([preprocess(q) for q, _ in FAQS])
    return vectorizer, matrix


def get_answer(user_question):
    cleaned = preprocess(user_question)
    if not cleaned:
        return FALLBACK
    vectorizer, matrix = build_model()
    scores = cosine_similarity(vectorizer.transform([cleaned]), matrix)[0]
    best = scores.argmax()
    if scores[best] < THRESHOLD:
        return FALLBACK
    return FAQS[best][1]


# ------------------------------------------------------------------
# 4) CHAT UI
# ------------------------------------------------------------------
st.set_page_config(page_title="Tech FAQ Chatbot", page_icon="🤖")
st.title("🤖 Tech FAQ Chatbot")
st.caption("Explore technology through conversation.")

with st.sidebar:
    st.header("Topics I know")
    for q, _ in FAQS:
        st.write("•", q)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state or not st.session_state.messages:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me about technology, like AI, coding, machine learning."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Type your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    answer = get_answer(prompt)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)