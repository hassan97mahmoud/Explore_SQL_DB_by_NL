## 💻 Tech Stack

The project is built with a modern Python technology stack:

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B9?style=for-the-badge&logo=google&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-AC1431?style=for-the-badge&logo=qdrant&logoColor=white)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-D71F00?style=for-the-badge&logo=sqlalchemy&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)

# Explore Your SQL Database with Natural Language

This project provides a powerful and intuitive web interface that allows users to interact with a SQL database using plain English. It leverages the power of Google's Gemini models to convert natural language questions into executable SQL queries, runs them against the database, and returns the answers in a conversational format.

![Demo GIF of the Application](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdWZnaGlmNWxhZnA4ZjhxN21qbzVvc215eG15ZGF2b3QzdHF0NmYwaCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/your-gif-id/giphy.gif)
*A brief demonstration of the application in action. 
---

## 🚀 Key Features

-   **Natural Language to SQL:** Converts complex user questions in plain English into accurate SQL queries.
-   **Dual Architecture:** Built with a robust **FastAPI** backend for logic and a user-friendly **Streamlit** frontend for interaction.
-   **Powered by Google Gemini:** Utilizes the `gemini-1.5-pro-latest` model for advanced language understanding and query generation.
-   **Few-Shot Learning:** Improves query generation accuracy by retrieving similar examples from a **Qdrant** vector database.
-   **Interactive UI:** Allows users to ask questions and receive answers in a simple, chat-like interface.
-   **Admin Dashboard:** Includes a separate admin view for testing the full API response and monitoring the backend health.

---

## 🛠️ How It Works

The application follows a logical pipeline to process a user's request:

1.  **User Input:** The user types a question into the Streamlit web interface (e.g., "How many customers are in Canada?").
2.  **API Request:** The Streamlit frontend sends the question to the FastAPI backend via an HTTP request.
3.  **Query Analysis (LLM Call #1):** The backend sends the question and database schema to Gemini to analyze relevance, rewrite the query for clarity, and identify the necessary tables.
4.  **Example Retrieval:** The backend uses the rewritten query to find similar question-SQL pairs from the Qdrant vector database.
5.  **SQL Generation (LLM Call #2):** A detailed prompt containing the schema, the similar examples, and the user's question is sent to Gemini to generate the final SQL query.
6.  **Database Execution:** The generated SQL is executed securely against the connected SQL database.
7.  **Natural Language Response (LLM Call #3):** The results from the database, along with the original question, are sent to Gemini to create a friendly, human-readable answer.
8.  **Display Answer:** The FastAPI backend sends the final answer back to the Streamlit frontend, which displays it to the user.

---

## ⚙️ Setup and Installation

Follow these steps to set up and run the project locally.

### 1. Clone the Repository
```bash
git clone https://github.com/hassan97mahmoud/Explore_SQL_DB_by_NL
cd Explore_SQL_DB_by_NL
2. Create and Activate a Virtual Environment
It's highly recommended to use a virtual environment.
code
Bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
Install all the required Python packages.
code
Bash
pip install -r requirements.txt
4. Set Up Environment Variables
Create a file named .env in the root of your project folder and add your credentials. Use the _template.env as a guide.
Important: The .env file is listed in .gitignore and should never be committed to version control.
code
Env
# .env file

# Get your key from Google AI Studio: https://makersuite.google.com/
GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"

# URL of your Qdrant instance (e.g., http://localhost:6333 or a cloud URL)
QDRANT_HOST="http://localhost:6333"
# API Key for Qdrant Cloud (leave empty if not using authentication)
QDRANT_API_KEY=""

# Database Connection String (SQLAlchemy format)
# Example for the included Chinook SQLite database
DB_CONNECTION_STRING="sqlite:///Chinook.db"

# Password for the Streamlit Admin UI
ADMIN_PASSWORD="admin"
5. Set Up Qdrant and the Database
Qdrant: The easiest way to run Qdrant is with Docker:
code
Bash
docker run -p 6333:6333 qdrant/qdrant
Database: This project is configured to use the Chinook.db sample database. Make sure you have this file in your project directory.
▶️ How to Run
You need to run the backend and frontend servers in two separate terminals.
1. Start the Backend Server (Terminal 1):
Navigate to the project directory and run:
code
Bash
python App1b.py
Wait until you see the message Uvicorn running on http://127.0.0.1:8000.
2. Start the Frontend Application (Terminal 2):
In a new terminal, navigate to the same project directory and run:
code
Bash
streamlit run App1f.py
This will open the web application in your browser.
📂 Project Structure
code
Code
Explore_SQL_DB_by_NL/
│
├── App1b.py            # The FastAPI backend server, logic, and API endpoints.
├── App1f.py            # The Streamlit frontend user interface.
│
├── .env                # Stores all secret keys and environment variables (Not committed).
├── requirements.txt    # A list of all Python dependencies for the project.
├── .gitignore          # Specifies which files Git should ignore.
└── README.md           # This file.
📄 License
This project is licensed under the MIT License. See the LICENSE file for more details.
📬 Contact
[Hassan Abu Esmael] - [hmaboesmael@gimal.com]
Project Link: https://github.com/hassan97mahmoud/Explore_SQL_DB_by_NL
