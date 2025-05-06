import os
from dotenv import load_dotenv
from typing import List, Dict
import numpy as np
import chromadb




load_dotenv()




class ChromaDB:
   def __init__(self):
       # Initialize with new simplified client
       self.client = chromadb.PersistentClient(path="./chroma_db")
       self.collection_name = os.getenv("CHROMA_COLLECTION_NAME", "youtube-qa")
       self._setup_collection()




   def _setup_collection(self):
       # Create or get the collection
       self.collection = self.client.get_or_create_collection(
           name=self.collection_name,
           metadata={"hnsw:space": "cosine"}  # Using cosine similarity
       )




   def store_transcriptions(self, transcriptions: List[Dict], embeddings: List[np.ndarray]):
       # Prepare data for ChromaDB
       ids = [f"vec_{idx}" for idx in range(len(transcriptions))]
       embeddings_list = [emb.tolist() for emb in embeddings]
       metadatas = [{
           "text": data["text"],
           "start": data["start"],
           "end": data["end"],
           "path": data.get("path", ""),
           "language": data.get("language", "en")
       } for data in transcriptions]
      
       # Add to collection
       self.collection.add(
           ids=ids,
           embeddings=embeddings_list,
           metadatas=metadatas
       )




   def search(self, query_embedding: np.ndarray, top_k: int = 3):
       # Convert numpy array to list
       query_embedding_list = query_embedding.tolist()
      
       # Query the collection
       results = self.collection.query(
           query_embeddings=[query_embedding_list],
           n_results=top_k,
           include=["metadatas", "distances"]
       )
      
       # Format results
       matches = []
       for i in range(top_k):
           if i < len(results["metadatas"][0]):
               matches.append({
                   "metadata": results["metadatas"][0][i],
                   "score": 1 - results["distances"][0][i]
               })
      
       return {"matches": matches}