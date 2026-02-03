from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0.1,
    google_api_key=os.environ.get("GOOGLE_API_KEY")
)
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)



from langchain_community.document_loaders import TextLoader




from langchain.text_splitter import RecursiveCharacterTextSplitter



from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
vectordb_file_path="H1bvisa_index"

def create_vector_db():
    loader = TextLoader(
    file_path=r"h1b_kbnowledge.txt",
    encoding="utf-8"
    )

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
    )

    docs = text_splitter.split_documents(documents)
    # Create FAISS vector database using CHUNKED documents
    vectordb = FAISS.from_documents(
        documents=docs,  
        embedding=embeddings
        )
    vectordb.save_local(vectordb_file_path)

    pass



def get_qa_chain():
    vectordb = FAISS.load_local(vectordb_file_path,embeddings,allow_dangerous_deserialization=True)

# Create retriever
    retriever = vectordb.as_retriever(
    search_kwargs={"k": 3}
    )
    
    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful chatbot that answers H-1B visa questions in a natural and conversational way.

        Rules:
        - Do NOT use bullet points or numbered lists
        - Do NOT say phrases like "based on the context" or "according to the provided data"
        - Answer in a short, clear paragraph
        - Use simple, human-friendly English
        - Prefer explaining in a calm and informative tone
        - Do not HALLUCINATE.

        Context usage rules:
        - If the answer IS clearly available in the context, use it directly.
        - If the answer is NOT explicitly found in the context, give a **generalized, high-level answer**
          based on common patterns mentioned in the context.
        - Do NOT invent statistics, numbers, or employer-specific claims that are not present.
        - Avoid absolute claims; use words like "generally", "often", or "tends to".

        CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER:
        """

    )
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
    )
    return rag_chain

if __name__ == "__main__":
    # create_vector_db()
    chain=get_qa_chain()
    response=chain.invoke("which occupation have less chances of approval ?")
    print(response)