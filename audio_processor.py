import os
import wave
import numpy as np
from typing import List, Dict
from pathlib import Path




def chunk_audio(
   audio_path: str,
   chunk_duration: float = 30.0,
   overlap: float = 5.0,
   output_dir: str = "audio_chunks"
) -> List[Dict]:
   """
   Split audio into chunks with overlap.
   Returns list of dictionaries with path, start, end.
   """
   Path(output_dir).mkdir(exist_ok=True)
   try:
       with wave.open(audio_path, 'rb') as wav_file:
           params = wav_file.getparams()
           # Keep original quality but reduce chunk size
           frames = wav_file.readframes(params.nframes)
           audio_data = np.frombuffer(frames, dtype=np.int16)
   except Exception as e:
       raise Exception(f"Failed to read WAV file: {str(e)}")
  
   sample_rate = params.framerate
   samples_per_chunk = int(chunk_duration * sample_rate)
   samples_overlap = int(overlap * sample_rate)
   chunks = []
  
   base_name = Path(audio_path).stem
   total_samples = len(audio_data)
   pointer = 0
  
   while pointer < total_samples:
       end_pointer = min(pointer + samples_per_chunk, total_samples)
       chunk_data = audio_data[pointer:end_pointer]
      
       chunk_path = os.path.join(
           output_dir,
           f"{base_name}_chunk_{len(chunks):03d}.wav"
       )
      
       with wave.open(chunk_path, 'wb') as chunk_file:
           chunk_file.setparams(params)
           chunk_file.writeframes(chunk_data.tobytes())
      
       start_time = pointer / sample_rate
       end_time = end_pointer / sample_rate
       chunks.append({
           "path": chunk_path,
           "start": start_time,
           "end": end_time
       })
      
       pointer += (samples_per_chunk - samples_overlap)
  
   return chunks