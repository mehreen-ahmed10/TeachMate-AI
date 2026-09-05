---
title: TeachMate AI
emoji: 🎓
colorFrom: yellow
colorTo: red
sdk: streamlit
app_file: app.py
pinned: false
---

# 🎓 TeachMate AI

**Your Intelligent Teaching Companion**

TeachMate AI is a GenAI-powered educational teaching assistant for teachers from Grade 1 to Grade 12.

## Features

- 📚 Lesson Planner
- 📝 Worksheet Generator
- 📊 Assessment Generator
- 🎯 Differentiated Learning
- 🔄 Remedial Learning
- 🎯 Learning Objectives
- 🤖 AI Teaching Assistant
- 📖 Curriculum-grounded RAG

## Subjects

English, Urdu, Mathematics, Science, History, Geography, Computer Science, Islamiyat.

## Technology

LangChain • ChatGroq • Pydantic • Chroma • Hugging Face Embeddings • Streamlit

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, `README.md`, `.gitignore`, and `.streamlit/config.toml`.
3. Go to Streamlit Community Cloud and choose **New app**.
4. Select your GitHub repository and `app.py`.
5. In **Settings → Secrets**, add:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

6. Deploy.

Never commit your Groq API key to GitHub.

## RAG storage note

The current implementation stores Chroma data in the local `teachmate_chroma` directory. Streamlit Community Cloud local storage is ephemeral, so curriculum uploads should not be considered permanent between restarts/redeployments. For production, use persistent external storage or a hosted vector database.
