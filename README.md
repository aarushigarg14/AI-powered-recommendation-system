# AI-Powered Furniture Recommendation & Analytics Platform
It is a full-stack AI application leveraging ML, NLP, CV, and GenAI to create an intelligent furniture discovery system. Users can search products via natural language queries and get semantically similar products with AI-generated descriptions.

## Features
- **Chat-Style Product Discovery**: Natural language search with conversational interface
- **AI-Generated Descriptions**: Creative product descriptions using GenAI
- **Semantic Search**: Vector database powered search using Pinecone
- **Analytics Dashboard**: Business insights with interactive charts
- **Dark Theme UI**: Modern navy/indigo design with responsive layout
- **Context-Aware Conversations**: Maintains chat history and context
- **Multi-Modal AI**: Combines text and image embeddings

## Tech Stack

### Backend
- **FastAPI**: High-performance API server
- **Pinecone**: Vector database for semantic search
- **LangChain**: GenAI integration
- **Transformers**: NLP models for embeddings
- **PyTorch**: Computer vision models

### Frontend  
- **React**: Modern UI framework
- **React Router**: Client-side routing
- **Tailwind CSS**: Utility-first styling
- **Dark Theme**: Navy/indigo gradient design


## Setup Instructions

### Prerequisites
- Python 3.8+
- Node.js 16+
- Git

### Backend Setup
```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
# i would prefer using py -3.11 venv venv
# this version of python is compatible with other libraries
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# you'll have to edit .env with your own API keys

python main_server.py


```

### Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

### Environment Variables
Create `.env` file in backend directory:
```env
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=your_pinecone_env
OPENAI_API_KEY=your_openai_key  # Optional
HUGGINGFACE_TOKEN=your_hf_token # Optional
```


## **Quick Start (4 Commands)**

```bash
# 1. Start Backend
cd backend 
venv\Scripts\activate
uvicorn main:app --reload

# 2. Start Frontend  
cd frontend 
npm start


