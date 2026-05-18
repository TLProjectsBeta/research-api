from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from app.chains.research import get_llm
from app.config import get_settings

settings = get_settings()

CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant. Help users explore topics deeply."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

base_chain = CHAT_PROMPT | get_llm() | StrOutputParser()

# In-memory store as fallback
memory_store = {}

def get_session_history(session_id: str):
    try:
        # Try Redis first
        return RedisChatMessageHistory(
            session_id=session_id,
            url=settings.redis_url,
            ttl=86400
        )
    except Exception:
        # Fall back to in-memory if Redis fails
        if session_id not in memory_store:
            memory_store[session_id] = ChatMessageHistory()
        return memory_store[session_id]

chat_chain = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)