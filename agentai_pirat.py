from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
import langchain
import os


def agentai_pirat(query: str, recreate_embeddings: bool = False) -> str:
    """
    Generate a greeting message for the user.

    Args:
        name (str): The name of the user.

    Returns:
        str: A personalized greeting message.
    """
    llm = OllamaLLM(model="llama3.2:1b", temperature=0)
    persist_directory = "chroma_db_dir"
    collection_name = "101800000001"
    embed_model = OllamaEmbeddings(model="llama3.2:1b")
    if recreate_embeddings or not os.path.exists(persist_directory):
        loader = PyMuPDFLoader("src/101800000001.pdf")
        data_pdf = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=500,
            length_function=len)
        docs = text_splitter.split_documents(data_pdf)
        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=embed_model,
            persist_directory=persist_directory,
            collection_name=collection_name)
    else:
        vector_store = Chroma(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_function=embed_model)

    retriever = vector_store.as_retriever(search_kwargs={'k': 3})
    custom_prompt_template = """Usa la siguiente información para responder a la pregunta del usuario. No debes inventar nada. \
    limitate a responder a la pregunta con el contexto que se te proporciona. Responde imitando a un pirata.
    
    Contexto: {context}
    Pregunta: {question}

    Respuesta útil:
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    qa = RetrievalQA.from_chain_type(llm=llm,
                                     chain_type="stuff",
                                     retriever=retriever,
                                     return_source_documents=True,
                                     chain_type_kwargs={"prompt": prompt})
    response = qa.invoke({"query": query})
    return response


def main():
    """
    Entry point of the program.
    """
    langchain.debug = True
    query = "Que es una facilidad de pago?"
    response = agentai_pirat(query)
    print(f"La respuesta es: {response['result']}")


if __name__ == "__main__":
    main()
