code
Markdown
# 📄 🗣️ Explore SQL DB with Natural Language

<div align="center">

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Cohere](https://img.shields.io/badge/Cohere-3755F9?style=for-the-badge&logo=cohere&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-AC1431?style=for-the-badge&logo=qdrant&logoColor=white)

</div>

This project provides a powerful and intuitive web interface that allows users to interact with any SQL database using plain English. It leverages the power of state-of-the-art Large Language Models (LLMs) to convert natural language questions into executable SQL queries, runs them against the database, and returns the answers in a polished, multi-component conversational format.

---

## 🎥 Live Demo

![Demo GIF of the Application](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdWZnaGlmNWxhZnA4ZjhxN21qbzVvc215eG15ZGF2b3QzdHF0NmYwaCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/your-gif-id/giphy.gif)
*A brief demonstration of the application's chat interface. **[Note: Replace the link above with a link to your own demo GIF or video!]***

---

## 🚀 Key Features

-   **Conversational Chat Interface:** Interact with your data through a familiar chat UI that remembers conversation history.
-   **Rich, Structured Responses:** Answers are displayed professionally with the main finding in bold, the raw data in an interactive table, and the generated SQL query in a collapsible section.
-   **Natural Language to SQL:** Converts complex user questions into accurate SQL queries.
-   **Dynamic Schema Loading:** Automatically inspects the connected database at startup, making the application adaptable to any SQL database without code changes.
-   **Dual Architecture:** Built with a robust **FastAPI** backend for logic and a user-friendly **Streamlit** frontend for interaction.
-   **Powered by Cohere:** Utilizes the `command-r-plus` model for advanced language understanding and high-quality query generation.
-   **Professional Debugging:** Integrated with **LangSmith** for end-to-end tracing and monitoring of the entire application pipeline.

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
| **LLM**              | ![Cohere](https://img.shields.io/badge/Cohere-3755F9?style=for-the-badge&logo=cohere&logoColor=white) (`command-r-plus`)                                                                              |
| **Embeddings**       | ![Cohere Embeddings](https://img.shields.io/badge/Cohere%20Embeddings-3755F9?style=for-the-badge&logo=cohere&logoColor=white) (`embed-english-v3.0`)                                                  |
| **Vector Store**     | ![Qdrant](https://img.shields.io/badge/Qdrant-AC1431?style=for-the-badge&logo=qdrant&logoColor=white)                                                                                                 |
| **Framework**        | ![LangChain](https://img.shields.io/badge/LangChain-008653?style=for-the-badge&logo=langchain&logoColor=white)                                                                                       |
| **Tracing**          | ![LangSmith](https://img.shields.io/badge/LangSmith-FD6801?style=for-the-badge&logo=langsmith&logoColor=white)                                                                                        |
| **Database Toolkit** | ![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-D71F00?style=for-the-badge&logo=sqlalchemy&logoColor=white)                                                                                     |

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

# Get your key from the Cohere Dashboard: https://dashboard.cohere.com/
COHERE_API_KEY="YOUR_COHERE_API_KEY_HERE"

# --- LangSmith Credentials ---
# Get these from https://smith.langchain.com/
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY="YOUR_LANGSMITH_API_KEY_HERE"
LANGCHAIN_PROJECT="Explore_SQL_DB_by_NL"

# --- Qdrant Vector Database Configuration ---
QDRANT_HOST="YOUR_QDRANT_HOST_URL"
QDRANT_API_KEY="YOUR_QDRANT_API_KEY"

# --- Database Connection String ---
DB_CONNECTION_STRING="sqlite:///Chinook.db"

# --- Streamlit Admin UI Password ---
ADMIN_PASSWORD="admin"
5. Set Up Qdrant and the Database
Qdrant: The easiest way to run Qdrant is with Docker: docker run -p 6333:6333 qdrant/qdrant
Database: This project is configured to use the Chinook.db sample database. Ensure this file is in your project directory.
▶️ How to Run
You need to run the backend and frontend servers in two separate terminals.
1. Start the Backend Server (Terminal 1):
code
Bash
# Corrected filename
python App3b.py
Wait until you see the message Uvicorn running on http://127.0.0.1:8000.
2. Start the Frontend Application (Terminal 2):
code
Bash
# Corrected filename
streamlit run App2f.py
This will open the web application in your browser.
📄 License
This project is licensed under the MIT License. See the LICENSE file for more details.
📬 Contact
Hassan Abu Esmael - hmaboesmael@gmail.com
Project Link: https://github.com/hassan97mahmoud/Explore_SQL_DB_by_NL