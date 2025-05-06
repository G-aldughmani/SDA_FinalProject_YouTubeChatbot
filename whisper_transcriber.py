import os
import time
from typing import List, Dict
import warnings
import numpy as np




# Suppress numpy warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")




try:
  import torch
  from transformers import pipeline
except ImportError as e:
  raise ImportError(f"Required packages not installed: {str(e)}")




class WhisperTranscriber:
  def __init__(self):
      """Initialize local Whisper-tiny transcriber"""
      # Verify numpy is working
      try:
          np.zeros(1)
      except Exception as e:
          raise RuntimeError(f"NumPy initialization failed: {str(e)}")
    
      # Initialize torch after numpy verification
      self.device = "cuda" if torch.cuda.is_available() else "cpu"
      self.model = "openai/whisper-tiny"
    
      try:
          self.pipe = pipeline(
              "automatic-speech-recognition",
              model=self.model,
              device=self.device,
              chunk_length_s=30,
              stride_length_s=[5, 3],
              return_timestamps=True
          )
          print(f"Device set to use {self.device}")
      except Exception as e:
          raise RuntimeError(f"Failed to initialize Whisper pipeline: {str(e)}")




  def transcribe_chunks(self, chunk_paths: List[Dict]) -> List[Dict]:
      """Transcribe audio chunks locally using Whisper-tiny"""
      results = []
      successful_chunks = 0




      for chunk in chunk_paths:
          try:
              # Verify file exists
              if not os.path.exists(chunk['path']):
                  print(f"File not found: {chunk['path']}")
                  continue




              # Transcribe with error handling
              output = self.pipe(
                  chunk['path'],
                  return_timestamps=True
              )




              # Process output
              if isinstance(output, dict) and "chunks" in output:
                  for segment in output["chunks"]:
                      results.append({
                          "text": segment["text"],
                          "start": chunk['start'] + segment["timestamp"][0],
                          "end": chunk['start'] + segment["timestamp"][1],
                          "path": chunk['path']
                      })
                  successful_chunks += 1
              else:
                  print(f"Unexpected output format from {chunk['path']}")




          except Exception as e:
              print(f"Failed to transcribe {chunk['path']}: {str(e)}")
              continue




      if not results:
          raise RuntimeError("No chunks were successfully transcribed")
        
      print(f"\nTranscription complete!")
      print(f"- Successfully transcribed {successful_chunks}/{len(chunk_paths)} chunks")
    
      return results