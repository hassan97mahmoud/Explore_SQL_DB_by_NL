# App1b.py (Backend - Corrected with latest model name)

# --- Imports ---
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Path
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
import uvicorn
import json
import uuid
import os
import shutil
from dotenv import load_dotenv

# LangChain and AI Imports
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.chains import LLMChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient, models
from langchain_core._api.deprecation import LangChainDeprecationWarning
import warnings
import asyncio

# --- Suppress specific warnings ---
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

# --- Configuration Loading ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "text2sql_examples")
DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")

# --- Global Client Initializations ---
llm = None
sql_generation_llm = None
natural_language_llm = None
embedding_model = None
qdrant_client_instance = None
vector_store = None
db = None

# --- Initialize models and services ---
if GOOGLE_API_KEY:
    asyncio.set_event_loop(asyncio.new_event_loop())
    # FIX: Updated model name to the latest stable version for Google AI Studio
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=GOOGLE_API_KEY, convert_system_message_to_human=True)
    sql_generation_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=GOOGLE_API_KEY, convert_system_message_to_human=True)
    natural_language_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=GOOGLE_API_KEY, convert_system_message_to_human=True)
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
else:
    print("WARNING: GOOGLE_API_KEY not set.")

if QDRANT_HOST and embedding_model:
    qdrant_client_instance = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)
    vector_store = Qdrant(client=qdrant_client_instance, collection_name=QDRANT_COLLECTION_NAME, embeddings=embedding_model)
else:
    print("WARNING: QDRANT_HOST not set or embedding model failed.")

if DB_CONNECTION_STRING:
    db = SQLDatabase.from_uri(DB_CONNECTION_STRING)
else:
    print("WARNING: DB_CONNECTION_STRING not set.")

# --- Database Schema and Prompt Templates ---
DB_SCHEMA_EXAMPLE = """
Album(AlbumId, Title, ArtistId)
Artist(ArtistId, Name)
Customer(CustomerId, FirstName, LastName, Company, Address, City, State, Country, PostalCode, Phone, Fax, Email, SupportRepId)
Employee(EmployeeId, LastName, FirstName, Title, ReportsTo, BirthDate, HireDate, Address, City, State, Country, PostalCode, Phone, Fax, Email)
Genre(GenreId, Name)
Invoice(InvoiceId, CustomerId, InvoiceDate, BillingAddress, BillingCity, BillingState, BillingCountry, BillingPostalCode, Total)
InvoiceLine(InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity)
MediaType(MediaTypeId, Name)
Playlist(PlaylistId, Name)
PlaylistTrack(PlaylistId, TrackId)
Track(TrackId, Name, AlbumId, MediaTypeId, GenreId, Composer, Milliseconds, Bytes, UnitPrice)
"""
DB_SCHEMA_EXAMPLE_DESCRIPTION = """
Album(AlbumId, Title, ArtistId) # Stores information about music albums.
Artist(ArtistId, Name) # Contains data about music artists.
Customer(CustomerId, FirstName, LastName, Company, Address, City, State, Country, PostalCode, Phone, Fax, Email, SupportRepId) # Holds customer information.
Employee(EmployeeId, LastName, FirstName, Title, ReportsTo, BirthDate, HireDate, Address, City, State, Country, PostalCode, Phone, Fax, Email) # Stores employee details.
Genre(GenreId, Name) # Represents music genres.
Invoice(InvoiceId, CustomerId, InvoiceDate, BillingAddress, BillingCity, BillingState, BillingCountry, BillingPostalCode, Total) # Contains invoice information.
InvoiceLine(InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) # Represents individual items within an invoice.
MediaType(MediaTypeId, Name) # Stores media types for tracks.
Playlist(PlaylistId, Name) # Contains user-created playlists.
PlaylistTrack(PlaylistId, TrackId) # Associates tracks with playlists.
Track(TrackId, Name, AlbumId, MediaTypeId, GenreId, Composer, Milliseconds, Bytes, UnitPrice) # Stores detailed information about each music track.
"""
RELEVANCE_REWRITE_TABLES_TYPES_PROMPT_TEMPLATE = """You are an AI assistant. Your task is to analyze a user question based on a database schema, determine if it's answerable, rewrite it for clarity, identify the relevant tables, and classify query types.
### Database Schema:
{schema}
### User Question:
{query}
### Instructions:
1. **Relevance**: Can the question be answered using the schema? ("yes", "no", "maybe").
2. **Rewrite the Question**: If relevant, rewrite the question to be clearer. If not relevant, return the original question.
3. **Identify Relevant Tables**: If relevant, list only the table names needed. Otherwise, return an empty list.
4. **Identify Query Types**: If relevant, list types of operations needed (e.g., `selection`, `filter`, `join`). Otherwise, return an empty list.
### Output Format:
Respond with a single, valid JSON object.
Example:
{{"relevant": "yes", "query": "Find the total number of customers in each country.", "relevant_tables": ["Customer"], "query_types": ["aggregation", "filter"]}}
### Now Process:
Database Schema:
{schema}
User Question:
{query}
Output JSON:
"""
TEXT_TO_SQL_INSTRUCTION = """You are an expert SQL generator. Given a database schema and a user question, generate a syntactically correct SQL query. Output ONLY the SQL query. Do not add any explanation or markdown formatting."""
SQL_RESULT_TO_NL_PROMPT_TEMPLATE = """You are an AI assistant. Given an original user question and the result of a SQL query, provide a concise, natural language response. Do not mention the SQL query.
Original User Question:
{user_question}
SQL Query Result:
{sql_result}
Natural Language Answer:
"""

# --- Pydantic Models (API Data Contracts) ---
class ProcessQueryRequest(BaseModel):
    user_question: str
class QueryAnalysisData(BaseModel):
    relevant: str
    query: str
    relevant_tables: List[str]
    query_types: List[str]
class SimilarExample(BaseModel):
    nl: str
    id: Optional[str] = None
    sql: Optional[str] = None
    tables: Optional[List[str]] = None
    type: Optional[str] = None
class ProcessQueryResponse(BaseModel):
    original_question: str
    analysis: Optional[QueryAnalysisData] = None
    similar_examples: List[SimilarExample] = []
    assembled_prompt_snippet: Optional[str] = None
    generated_sql: Optional[str] = None
    query_result: Optional[Any] = None
    nl_response: Optional[str] = None
    error_message: Optional[str] = None
class NLSQLInputExample(BaseModel):
    nl: str = Field(...)
    sql: Optional[str] = None
    tables: Optional[List[str]] = None
    type: Optional[str] = None
    id: Optional[str] = None
class QdrantPoint(BaseModel):
    id: Any
    payload: Optional[Dict[str, Any]] = None
    vector: Optional[Any] = None
class GetAllPointsResponse(BaseModel):
    points: List[QdrantPoint]
    next_offset: Optional[Any] = None
    count: int

TEMP_UPLOAD_DIR = "temp_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

# --- Core Logic Functions ---
def validate_rewrite_identify_tables_and_types_logic(user_query: str, db_schema: str, llm_instance) -> str:
    prompt = PromptTemplate(template=RELEVANCE_REWRITE_TABLES_TYPES_PROMPT_TEMPLATE, input_variables=["query", "schema"])
    chain = LLMChain(llm=llm_instance, prompt=prompt)
    response = chain.invoke({"query": user_query, "schema": db_schema})
    return response['text']
def retrieve_similar_examples_logic(query_text: str, vector_store_instance, k: int = 3) -> list:
    if not query_text or not query_text.strip() or vector_store_instance is None: return []
    try:
        similar_docs = vector_store_instance.similarity_search(query_text, k=k)
        return [{"nl": doc.page_content, **doc.metadata} for doc in similar_docs]
    except Exception: return []
def format_dynamic_schema_logic(relevant_table_names: list, full_db_schema: str) -> str:
    if not relevant_table_names: return "No specific table schema provided."
    schema_lines = full_db_schema.strip().split('\n')
    dynamic_schema_parts = []
    for table_name in relevant_table_names:
        for line_idx, line in enumerate(schema_lines):
            if line.strip().startswith(table_name + "("):
                dynamic_schema_parts.append(line.strip())
                for desc_line_idx in range(line_idx + 1, len(schema_lines)):
                    desc_line = schema_lines[desc_line_idx].strip()
                    if desc_line.startswith("#"): dynamic_schema_parts.append(desc_line)
                    elif "(" in desc_line and ")" in desc_line and not desc_line.startswith("#"): break
                    elif not desc_line.startswith("#") and desc_line: break
                break
    return "\n".join(dynamic_schema_parts) if dynamic_schema_parts else "Selected table schemas not found."
def format_few_shot_examples_logic(few_shot_examples: list) -> str:
    if not few_shot_examples: return ""
    formatted_examples_str = "### Examples (NL to SQL):\n"
    for ex in few_shot_examples:
        nl, sql = ex.get('nl', 'N/A'), ex.get('sql', 'N/A')
        formatted_examples_str += f"-- User Question: {nl}\nSQL: {sql}\n\n"
    return formatted_examples_str.strip()
def assemble_text_to_sql_prompt_logic(instruction: str, rewritten_query: str, few_shot_examples: list, relevant_table_names: list, full_db_schema: str) -> str:
    dynamic_schema_str = format_dynamic_schema_logic(relevant_table_names, full_db_schema)
    few_shots_str = format_few_shot_examples_logic(few_shot_examples)
    prompt_parts = [instruction, "\n### Database Schema:", dynamic_schema_str]
    if few_shots_str: prompt_parts.append("\n" + few_shots_str)
    prompt_parts.extend(["\n### Task:", f"User Question: {rewritten_query}", "SQL Query:"])
    return "\n".join(prompt_parts)
def generate_sql_from_prompt_logic(assembled_prompt: str, sql_llm_instance) -> str:
    prompt_template = PromptTemplate.from_template("{final_prompt}")
    sql_generation_chain = LLMChain(llm=sql_llm_instance, prompt=prompt_template)
    response = sql_generation_chain.invoke({"final_prompt": assembled_prompt})
    sql_query = response.get('text', '').strip()
    if sql_query.lower().startswith("```sql"): sql_query = sql_query[6:].strip()
    if sql_query.endswith("```"): sql_query = sql_query[:-3].strip()
    return sql_query
def execute_sql_query_logic(sql_query: str, db_instance):
    if not db_instance: raise HTTPException(status_code=500, detail="Database connection not available.")
    if not sql_query or not sql_query.strip(): return "No SQL query to execute."
    try: return db_instance.run(sql_query)
    except Exception as e: return f"Error executing SQL: {str(e)}"
def generate_natural_language_response_logic(user_question: str, sql_result: str, nl_llm_instance) -> str:
    prompt = PromptTemplate(template=SQL_RESULT_TO_NL_PROMPT_TEMPLATE, input_variables=["user_question", "sql_result"])
    chain = LLMChain(llm=nl_llm_instance, prompt=prompt)
    response = chain.invoke({"user_question": user_question, "sql_result": str(sql_result)})
    return response.get('text', "Could not generate a natural language response.").strip()

# --- FastAPI Application and Endpoints ---
api = FastAPI(title="Text2SQL Backend API")

@api.get("/")
def root():
    return {"message": "Welcome to the Text2SQL API. See /docs for details."}

@api.get("/health")
def health_check():
    return {"status": "ok", "database_status": "connected" if db else "not connected", "vector_store_status": "available" if vector_store else "not available", "llm_status": "available" if llm else "not available"}

@api.post("/process-query", response_model=ProcessQueryResponse)
def process_query_endpoint(request: ProcessQueryRequest):
    user_question = request.user_question
    response_data = ProcessQueryResponse(original_question=user_question)
    try:
        llm_output_json_str = validate_rewrite_identify_tables_and_types_logic(user_question, DB_SCHEMA_EXAMPLE, llm)
        first_brace, last_brace = llm_output_json_str.find('{'), llm_output_json_str.rfind('}')
        if first_brace != -1 and last_brace > first_brace: extracted_json_str = llm_output_json_str[first_brace : last_brace + 1]
        else: raise ValueError("LLM did not return a valid JSON object for analysis.")
        analysis_dict = json.loads(extracted_json_str)
        response_data.analysis = QueryAnalysisData(**analysis_dict)
        if response_data.analysis.relevant in ['yes', 'maybe']:
            similar_examples_raw = retrieve_similar_examples_logic(response_data.analysis.query, vector_store)
            response_data.similar_examples = [SimilarExample(**ex) for ex in similar_examples_raw]
            final_prompt = assemble_text_to_sql_prompt_logic(TEXT_TO_SQL_INSTRUCTION, response_data.analysis.query, [ex.dict() for ex in response_data.similar_examples], response_data.analysis.relevant_tables, DB_SCHEMA_EXAMPLE_DESCRIPTION)
            response_data.assembled_prompt_snippet = final_prompt[:1000]
            generated_sql = generate_sql_from_prompt_logic(final_prompt, sql_generation_llm)
            response_data.generated_sql = generated_sql
            if generated_sql and db:
                query_result = execute_sql_query_logic(generated_sql, db)
                response_data.query_result = str(query_result)
                if "Error" not in str(query_result):
                    nl_response = generate_natural_language_response_logic(user_question, str(query_result), natural_language_llm)
                    response_data.nl_response = nl_response
                else: response_data.nl_response = "I encountered an error while running the SQL query."
            else: response_data.nl_response = "I was able to generate a SQL query but couldn't execute it."
        else: response_data.nl_response = "This question does not seem to be answerable with the available data."
    except Exception as e: response_data.error_message = f"An unexpected error occurred: {str(e)}"
    return response_data

@api.get("/get-all-examples", response_model=GetAllPointsResponse)
def get_all_examples_endpoint(limit: int = 10, offset: Optional[str] = None):
    if not qdrant_client_instance: raise HTTPException(status_code=503, detail="Qdrant client not available.")
    try:
        points_data, next_offset = qdrant_client_instance.scroll(collection_name=QDRANT_COLLECTION_NAME, limit=limit, offset=offset, with_payload=True, with_vectors=False)
        result_points = [{"id": p.id, "payload": p.payload} for p in points_data]
        return GetAllPointsResponse(points=[QdrantPoint(**p) for p in result_points], next_offset=str(next_offset) if next_offset else None, count=len(result_points))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Main Execution Block for Backend ---
if __name__ == "__main__":
    print("--- Starting FastAPI Server ---")
    uvicorn.run("App1b:api", host="127.0.0.1", port=8000, reload=True)