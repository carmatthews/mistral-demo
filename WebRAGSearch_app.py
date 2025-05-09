import os  
from dotenv import load_dotenv

import streamlit as st  
import httpx
import time

from langchain_mistralai import ChatMistralAI

# WebSearch Libraries
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage  
from langchain_core.prompts import ChatPromptTemplate
from tavily import TavilyClient   

# RAG Libraries
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain 
import glob


load_dotenv()

mistral_api_key = os.getenv("MISTRAL_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

PDF_FOLDER = "docs"

# Mistral LLM  
llm = ChatMistralAI(  
    api_key=mistral_api_key,  
    model="mistral-large-latest",  
    temperature=0.8,  
)  

##### Web Search #####
@tool
def tavily_search_tool(search_query: str, num_results: int = 3) -> str: 
    """Search the web using Tavily API.

    The search results return:
        title: The title of the search result
        url: The URL of the search result
        content: The most relevant content extracted from the URL based on the query
        score: A relevance score for the search result
    """ 
    client = TavilyClient(api_key=tavily_api_key)  
    results = client.search(query=search_query, max_results=num_results) 

    formatted = "\n\n".join(  
        [f"{i+1}. {item['title']}\n{item['url']}\n{item['content']}" for i, item in enumerate(results["results"])]  
    )  
    return formatted  

SEARCH_ENGINE_MSG = """
Analyze the question then use the tavily_search_tool to search the web for the answer and find the best 10 results then provide a one sentence answer to the user.  

Only call the tavily_search_tool tool ONCE.  

When you get the results, make certain you found the answer to the user's question given the context of the conversation.

If you have a specific answer, then summarize the information in a single sentence and return the links for the 10 search results as well.

If you are not able to find an answer for the user's question, just state "I am sorry, I couldn't find an answer on the Web for this.  Please try the Archive instead.".
"""

# Setup the Graph  
llm_with_tools = llm.bind_tools([tavily_search_tool])  
  
def llm_node(state: MessagesState):  
    search_prompt = ChatPromptTemplate.from_messages(  
        [("system", SEARCH_ENGINE_MSG), ("human", "{messages}")]  
    )  
    search_for_answers = search_prompt | llm_with_tools  
    messages = state["messages"]  
    query_response = search_for_answers.invoke({"messages": messages})  
    return {"messages": [query_response]}  

def tool_node(state: MessagesState):  
    tool_calls = state["messages"][-1].tool_calls  
    tools_mapping = {tavily_search_tool.name: tavily_search_tool}  
    results = []  
    for tool_call in tool_calls:  
        tool = tools_mapping.get(tool_call["name"])  
        if not tool:  
            output = "bad tool name, retry"  
        else:  
            output = tool.invoke(tool_call["args"])  
        results.append(  
            ToolMessage(  
                tool_call_id=tool_call["id"],  
                name=tool_call["name"],  
                content=str(output),  
            )  
        )  
    state["messages"] = results  
    return state  

def exists_action(state: MessagesState):  
    result = state['messages'][-1]  
    return len(result.tool_calls) > 0  

# Build the Graph
graph_builder = StateGraph(MessagesState)  
graph_builder.add_node("llm_mistral", llm_node)  
graph_builder.add_node("tool", tool_node)  
graph_builder.add_edge(START, "llm_mistral")  
graph_builder.add_conditional_edges(  
    "llm_mistral",  
    exists_action,  
    {True: "tool", False: END}  
)  
graph_builder.add_edge("tool", "llm_mistral")  
search_agent = graph_builder.compile()  

##### RAG Search #####
@st.cache_resource  
def load_and_split_pdfs(pdf_folder):  
    docs = []  
    for path in glob.glob(f"{pdf_folder}/*.pdf"):  
        print(f"Loading {path}")
        loader = PyPDFLoader(path)  
        docs.extend(loader.load())  
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)  
    split_docs = splitter.split_documents(docs)  
    print(f"Loaded {len(split_docs)} documents from {PDF_FOLDER} folder.")
    return split_docs  

@st.cache_resource  
def get_vector_store(_split_docs):  
    print(f"Creating embeddings for {len(_split_docs)} documents")
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    vectorstore = FAISS.from_documents(_split_docs, embeddings)  
    print(f"Created FAISS index with {len(_split_docs)} documents.")
    return vectorstore 

@st.cache_resource  
def get_rag_chain(_vectorstore):  
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 4})  
    chain = ConversationalRetrievalChain.from_llm(llm, retriever)  
    print("RAG chain created.")
    return chain  

def initialize_rag_chain():  
    split_docs = load_and_split_pdfs(PDF_FOLDER)  
    vectorstore = get_vector_store(_split_docs=split_docs)  
    rag_chain = get_rag_chain(_vectorstore=vectorstore)  
    return rag_chain    

def format_web_result(result_msg):  
    """  
    Formats the web search result for display.  
    Handles both plain string and structured list-of-dict outputs.  
    """  
    if isinstance(result_msg, str):  
        return result_msg  
  
    # If it's a list, try to pretty-print  
    if isinstance(result_msg, list):  
        text_parts = []  
        refs = {}  
        for item in result_msg:  
            if item.get("type") == "text":  
                text_parts.append(item.get("text", ""))  
            elif item.get("type") == "reference" and "reference_ids" in item:  
                refs.update({rid: None for rid in item["reference_ids"]})  
        # Just join text parts for now  
        return "\n".join(text_parts).strip()  
  
    # Fallback: just str()  
    return str(result_msg)  

def main():  
    st.title("🔎📚 Archive & Web Search Chat")  
  
    # --- Session State ---  
    if "history" not in st.session_state:  
        st.session_state["history"] = []  # List of (speaker, message) tuples  
  
    if "rag_chain" not in st.session_state:  
        try:  
            st.session_state["rag_chain"] = initialize_rag_chain()  
        except Exception as e:  
            st.error(f"Failed to initialize RAG chain: {e}")  
            return  
  
    # --- User Input ---  
    user_query = st.text_input("Ask a question (about your PDFs or from the web):", key="input")  
  
    col1, col2, col3 = st.columns([1, 1, 1])  
    with col1:  
        ask_web = st.button("Ask the Web")  
    with col2:  
        ask_archive = st.button("Ask the Archive")  
    with col3:  
        clear = st.button("🧹 Clear chat history")  
  
    if clear:  
        st.session_state.history = []  
  
    # --- Action Handlers ---  
    if user_query and (ask_archive or ask_web):  
        st.session_state.history.append(("You", user_query))  

        # ---- Build conversation history for both agents ----  
        # For RAG (archive), tuple list of (user, ai) turns  
        rag_chat_history = []  
        last_user = None  
        for speaker, msg in st.session_state.history:  
            if speaker == "You":  
                last_user = msg  
            elif speaker in ("Archive", "Web") and last_user is not None:  
                # Pair user with last AI reply  
                rag_chat_history.append((last_user, msg))  
                last_user = None  
    
        # For Web, LangChain message objects  
        web_messages = []  
        for speaker, msg in st.session_state.history:  
            if speaker == "You":  
                web_messages.append(HumanMessage(content=msg))  
            elif speaker == "Archive":  
                web_messages.append(AIMessage(content=msg))  
            elif speaker == "Web":  
                web_messages.append(AIMessage(content=msg))  
        # Add current user query if not already included  
        if not web_messages or not isinstance(web_messages[-1], HumanMessage):  
            web_messages.append(HumanMessage(content=user_query))  
        
        if ask_archive:  
            print(f"Asking the archive...{user_query}")
            # --- Archive/RAG Search ---  
            for attempt in range(2):  
                try:  
                    rag_chain = st.session_state["rag_chain"]  
                    # Pass the full chat history and the most recent question
                    result = rag_chain.invoke({  
                        "question": user_query,  
                        "chat_history": rag_chat_history  
                    })  
                    answer = result.get("answer", "No answer found.")  
                    st.session_state.history.append(("Archive", answer))  
                    break  
                except httpx.HTTPStatusError as e:  
                    if e.response.status_code == 429:  
                        if attempt == 0:  
                            st.warning("Rate limit hit; retrying in 5 seconds...")  
                            time.sleep(5)  
                        else:  
                            st.error("🚦 Rate limit reached again. Please wait and try later.")  
                            st.session_state.history.append(("Archive", "Error: Rate limit reached."))  
                    else:  
                        st.error(f"An error occurred: {e}")  
                        st.session_state.history.append(("Archive", f"Error: {e}"))  
                        break  
                except Exception as e:  
                    st.error(f"An unexpected error occurred: {e}")  
                    st.session_state.history.append(("Archive", f"Error: {e}"))  
                    break  
  
        elif ask_web:  
            print(f"Asking the web...{user_query}")
            for attempt in range(2):  
                try:
                    # Pass the full conversation as messages  
                    result = search_agent.invoke({"messages": web_messages})  
                    result_msg = result['messages'][-1].content  
                    formatted_msg = format_web_result(result_msg)  
                    st.session_state.history.append(("Web", formatted_msg))  
                    break  
                except httpx.HTTPStatusError as e:  
                    if e.response.status_code == 429:  
                        if attempt == 0:  
                            st.warning("Rate limit hit; retrying in 5 seconds...")  
                            time.sleep(5)  
                        else:  
                            st.error("🚦 Rate limit reached again. Please wait and try later.")  
                            st.session_state.history.append(("Web", "Error: Rate limit reached."))  
                    else:  
                        st.error(f"An error occurred: {e}")  
                        st.session_state.history.append(("Web", f"Error: {e}"))  
                        break  
                except Exception as e:  
                    st.error(f"Error: {e}")  
                    st.session_state.history.append(("Web", f"Error: {e}"))  
                    break  


    # --- Conversation Display ---  
    st.markdown("---")  
    st.markdown("### Conversation")  
    for i, (speaker, msg) in enumerate(st.session_state.history):  
        if speaker == "You":  
            st.markdown(f"**You:** {msg}")  
        elif speaker == "Archive":  
            st.markdown(f"**📚 Archive:** {msg}")  
        elif speaker == "Web":  
            st.markdown(f"**🌐 Web:** {msg}")  
        else:  
            st.markdown(f"**{speaker}:** {msg}") 
 
if __name__ == "__main__":  
    main()  

