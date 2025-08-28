# 📄 🗣️ Explore SQL DB with Natural Language

<div align="center">

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B9?style=for-the-badge&logo=google&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-AC1431?style=for-the-badge&logo=qdrant&logoColor=white)

</div>

This project provides a powerful and intuitive web interface that allows users to interact with a SQL database using plain English. It leverages the power of Google's Gemini models to convert natural language questions into executable SQL queries, runs them against the database, and returns the answers in a conversational format.

---

## 🚀 Key Features

-   **Natural Language to SQL:** Converts complex user questions in plain English into accurate SQL queries.
-   **Dual Architecture:** Built with a robust **FastAPI** backend for logic and a user-friendly **Streamlit** frontend for interaction.
-   **Powered by Google Gemini:** Utilizes the `gemini-1.5-pro-latest` model for advanced language understanding and query generation.
-   **Few-Shot Learning:** Improves query generation accuracy by retrieving similar examples from a **Qdrant** vector database.
-   **Interactive UI:** Allows users to ask questions and receive answers in a simple, chat-like interface.
-   **Admin Dashboard:** Includes a separate admin view for testing the full API response and monitoring the backend health.

## 🧠 Use Case Examples

Ask complex questions just like you would to a data analyst:

-   *“How many customers are there in the USA?”*
-   *“Show me the top 5 most expensive tracks.”*
-   *“List all albums by the band 'AC/DC'.”*
-   *“What is the total revenue for invoice #10?”*
-   *“Which employee has the most customers reporting to them?”*

## 🛠️ Tech Stack

| Layer                | Technology                                                                                                                                                                                          |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Backend**          | ![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)                                                                                                             |
| **Frontend**         | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)                                                                                       |
| **LLM**              | ![Google Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B9?style=for-the-badge&logo=google&logoColor=white) (`gemini-1.5-pro-latest`)                                                     |
| **Embeddings**       | ![Google Generative AI Embeddings](https://img.shields.io/badge/Google%20Embeddings-4285F4?style=for-the-badge&logo=google&logoColor=white)                                                          |
| **Vector Store**     | ![Qdrant](https://img.shields.io/badge/Qdrant-AC1431?style=for-the-badge&logo=qdrant&logoColor=white)                                                                                                 |
| **Framework**        | ![LangChain](https://img.shields.io/badge/LangChain-008653?style=for-the-badge&logo=langchain&logoColor=white)                                                                                       |
| **Database Toolkit** | ![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-D71F00?style=for-the-badge&logo=sqlalchemy&logoColor=white)                                                                                     |
| **Deployment**       | ![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white) (for Qdrant)                                                                             |

## 📁 Getting Started

Follow these steps to set up and run the project locally.

### 1. Clone the Repository
```bash
git clone https://github.com/hassan97mahmoud/Explore_SQL_DB_by_NL.git
cd Explore_SQL_DB_by_NL
2. Create and Activate a Virtual Environment
code
Bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
code
Bash
pip install -r requirements.txt
4. Set Up Environment Variables
Create a file named .env in the root of your project folder. This file stores your secret credentials and is ignored by Git.
code
Env
# .env file

# Get your key from Google AI Studio: https://makersuite.google.com/
GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"

# URL of your Qdrant instance (e.g., http://localhost:6333)
QDRANT_HOST="http://localhost:6333"
QDRANT_API_KEY=""

# Database Connection String
DB_CONNECTION_STRING="sqlite:///Chinook.db"

# Password for the Streamlit Admin UI
ADMIN_PASSWORD="admin"
5. Set Up Qdrant and the Database
Qdrant: The easiest way to run Qdrant is with Docker:
code
Bash
docker run -p 6333:6333 qdrant/qdrant
Database: This project is configured to use the Chinook.db sample database. Make sure this file is in your project directory.
▶️ How to Run
You need to run the backend and frontend servers in two separate terminals.
1. Start the Backend Server (Terminal 1):
code
Bash
python App1b.py
Wait until you see the message Uvicorn running on http://127.0.0.1:8000.
2. Start the Frontend Application (Terminal 2):
code
Bash
streamlit run App1f.py
This will open the web application in your browser.
📄 License
This project is licensed under the MIT License. See the LICENSE file for more details.
📬 Contact
Hassan Abu Esmael - hmaboesmael@gmail.com
Project Link: https://github.com/hassan97mahmoud/Explore_SQL_DB_by_NL