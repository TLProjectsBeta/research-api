from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from app.config import get_settings

settings = get_settings()

RESEARCH_SYSTEM_PROMPT = """You are an expert research assistant. When given a topic,
provide a structured research summary with these sections:

## Overview
Brief explanation of the topic (2-3 sentences)

## Key Concepts
3-5 core concepts with brief explanations

## Current Applications
Real-world applications and use cases

## Key Players / Resources
Companies, researchers, or resources relevant to this topic

## Future Outlook
Where this field is heading

Keep each section concise and factual."""

RESEARCH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RESEARCH_SYSTEM_PROMPT),
    ("human", "Research topic: {topic}\n\nFocus area (optional): {focus}")
])

def get_llm():
    """Return LLM based on config. Swap models by changing .env, not code."""
    if settings.google_api_key:
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=settings.temperature,
            max_output_tokens=settings.max_tokens,
            google_api_key=settings.google_api_key
        )
    return ChatOpenAI(
        model=settings.model_name,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        openai_api_key=settings.openai_api_key,
        streaming=True
    )

# The main runnable — LangServe exposes this
research_chain = (
    RESEARCH_PROMPT
    | get_llm()
    | StrOutputParser()
)