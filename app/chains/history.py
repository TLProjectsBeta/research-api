from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from app.chains.research import get_llm
from app.config import get_settings

settings = get_settings()

CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant. Help users explore topics deeply."),
    MessagesPlaceholder(variable_name="history"),  # chat history injected here
    ("human", "{question}")
])

base_chain = CHAT_PROMPT | get_llm() | StrOutputParser()

def get_redis_history(session_id: str) -> RedisChatMessageHistory:
    return RedisChatMessageHistory(
        session_id=session_id,
        url=settings.redis_url,
        ttl=86400  # 24 hours TTL — old sessions auto-cleaned
    )

chat_chain = RunnableWithMessageHistory(
    base_chain,
    get_redis_history,
    input_messages_key="question",
    history_messages_key="history"
)