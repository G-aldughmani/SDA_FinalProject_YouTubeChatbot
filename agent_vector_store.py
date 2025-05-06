import os
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate


class VideoQAAgent:
   def __init__(self):
       # Hardcoded API key
       self.api_key = "your-api-key"


       # Initialize components with error handling
       try:
           self.llm = ChatOpenAI(
               temperature=0.3,
               openai_api_key=self.api_key,
               model="gpt-3.5-turbo-16k",  # Using 16k context for longer videos
               max_tokens=2000
           )
          
           self.embeddings = OpenAIEmbeddings(
               openai_api_key=self.api_key,
               model="text-embedding-3-small"
           )
          
           self.text_splitter = RecursiveCharacterTextSplitter(
               chunk_size=1500,
               chunk_overlap=200,
               length_function=len
           )
          
           self._setup_qa_system()
          
       except Exception as e:
           print(f"❌ Initialization failed: {str(e)}")
           raise


   def _setup_qa_system(self):
       """Initialize the QA system with proper error handling"""
       try:
           # Load and process the transcription file
           if not os.path.exists("transcription.txt"):
               raise FileNotFoundError("transcription.txt not found")
          
           loader = TextLoader("transcription.txt")
           documents = loader.load()
          
           if len(documents) == 0:
               raise ValueError("Empty transcription file")
          
           # Split text into manageable chunks
           texts = self.text_splitter.split_documents(documents)
          
           # Create vector store
           self.vector_store = Chroma.from_documents(
               documents=texts,
               embedding=self.embeddings,
               persist_directory="./chroma_db"
           )
          
           # Create optimized prompt template
           template = """
           You are a helpful AI assistant that answers questions about YouTube videos.
           Use the following transcript excerpts to answer the question.
           Include timestamps like [00:12:34] when relevant.
           If you don't know the answer, say "I don't know".
          
           Context: {context}
          
           Question: {question}
           Answer:"""
          
           QA_PROMPT = PromptTemplate(
               template=template,
               input_variables=["context", "question"]
           )
          
           # Initialize QA chain
           self.qa_chain = RetrievalQA.from_chain_type(
               llm=self.llm,
               chain_type="stuff",
               retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
               chain_type_kwargs={"prompt": QA_PROMPT},
               return_source_documents=True
           )
          
       except Exception as e:
           print(f"❌ Failed to setup QA system: {str(e)}")
           raise


   def ask_question(self, question: str) -> str:
       """Handle Q&A with proper error handling"""
       try:
           if not question.strip():
               return "Please enter a valid question"
          
           # Limit question length
           if len(question) > 500:
               return "Question too long (max 500 characters)"
          
           # Process the question
           result = self.qa_chain.invoke({"query": question})
           answer = result.get("result", "No answer found")
          
           # Add timestamp references if available
           if "source_documents" in result:
               timestamps = []
               for doc in result["source_documents"]:
                   if hasattr(doc, 'metadata'):
                       if 'start' in doc.metadata and 'end' in doc.metadata:
                           timestamps.append(f"[{doc.metadata['start']:.1f}-{doc.metadata['end']:.1f}s")
              
               if timestamps:
                   answer += f"\n\n(References: {', '.join(timestamps)})"
          
           return answer
          
       except Exception as e:
           return f"⚠️ Error processing question: {str(e)}"


   def cleanup(self):
       """Clean up resources"""
       try:
           if hasattr(self, 'vector_store'):
               self.vector_store.delete_collection()
       except Exception as e:
           print(f"Cleanup error: {str(e)}")