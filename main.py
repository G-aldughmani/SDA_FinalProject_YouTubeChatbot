#!/usr/bin/env python3
"""
YouTube Video Transcription and Q&A System v2.0
- Enhanced downloader with cookie support
- Local Whisper transcription
- Interactive Q&A
- LangSmith integration
"""


import os
import time
import random
from pathlib import Path
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from agent_vector_store import VideoQAAgent
from whisper_transcriber import WhisperTranscriber
from audio_processor import chunk_audio
import yt_dlp




# Load environment variables
load_dotenv()




# Initialize LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = "YouTube-QA-Chatbot-l"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"




class YouTubeDownloader:
  """Enhanced YouTube audio downloader with cookie support and fallbacks"""
  def __init__(self):
      self.user_agents = [
          'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
          'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
          'Mozilla/5.0 (X11; Linux x86_64)',
          'Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X)'
      ]
      self.referers = [
          'https://www.google.com/',
          'https://www.facebook.com/',
          'https://www.reddit.com/',
          'https://www.twitter.com/'
      ]




  def get_cookies(self):
      """Extract browser cookies for YouTube"""
      try:
          import browser_cookie3
          return browser_cookie3.load(domain_name='youtube.com')
      except Exception as e:
          print(f"⚠️ Couldn't load browser cookies: {str(e)}")
          return None




  def get_ydl_options(self):
      """Generate download options with randomized headers"""
      return {
          'format': 'bestaudio/best',
          'postprocessors': [{
              'key': 'FFmpegExtractAudio',
              'preferredcodec': 'wav',
              'preferredquality': '5',
          }],
          'outtmpl': 'audio_downloads/%(title)s.%(ext)s',
          'quiet': False,
          'no_warnings': False,
          'retries': 10,
          'fragment_retries': 10,
          'extractor_retries': 3,
          'http_headers': {
              'User-Agent': random.choice(self.user_agents),
              'Referer': random.choice(self.referers),
              'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
              'Accept-Language': 'en-US,en;q=0.5',
              'Accept-Encoding': 'gzip, deflate',
          },
          'socket_timeout': 30,
          'extract_flat': True,
          'force_generic_extractor': True,
          'noplaylist': True,
          'cookiefile': self.get_cookies(),
          'verbose': True
      }




  def download_with_ytdlp(self, video_url: str) -> Tuple[str, str]:
      """Primary download method with yt-dlp"""
      try:
          with yt_dlp.YoutubeDL(self.get_ydl_options()) as ydl:
              info = ydl.extract_info(video_url, download=True)
              file_path = ydl.prepare_filename(info).replace('.webm', '.wav').replace('.m4a', '.wav')
              return file_path, info.get('title', 'untitled')
      except Exception as e:
          raise Exception(f"yt-dlp failed: {str(e)}")




  def download_with_pytube(self, video_url: str) -> Tuple[str, str]:
      """Fallback method using pytube"""
      try:
          from pytube import YouTube
          yt = YouTube(video_url,
                      use_oauth=True,
                      allow_oauth_cache=True)
          stream = yt.streams.filter(only_audio=True).first()
          output_path = stream.download(output_path='audio_downloads')
          return output_path, yt.title
      except Exception as e:
          raise Exception(f"pytube failed: {str(e)}")




def download_youtube_audio(video_url: str) -> Tuple[str, str]:
  """Main download function with multiple fallbacks"""
  downloader = YouTubeDownloader()
  methods = [
      downloader.download_with_ytdlp,
      downloader.download_with_pytube
  ]
  for attempt in range(3):  # FIXED indentation here
      for method in methods:
          try:
              return method(video_url)
          except Exception as e:
              print(f"Attempt {attempt + 1} with {method.__name__} failed: {str(e)}")
              time.sleep(random.randint(2, 5))
  raise Exception("All download methods failed after 3 attempts each")




def save_transcription(transcriptions: List[Dict], filename: str = "transcription.txt"):
  """Save transcription segments to a text file with timestamps"""
  try:
      with open(filename, 'w', encoding='utf-8') as f:
          for segment in transcriptions:
              f.write(f"[{segment['start']:.2f}-{segment['end']:.2f}] {segment['text']}\n")
      print(f"✓ Transcription saved to {filename}")
  except Exception as e:
      print(f"Failed to save transcription: {str(e)}")
      raise




def transcribe_audio(url: str) -> bool:
  """Complete audio transcription pipeline"""
  try:
      # Create necessary directories
      os.makedirs("audio_downloads", exist_ok=True)
      os.makedirs("audio_chunks", exist_ok=True)




      # Download audio with multiple fallbacks
      audio_path = "audio_downloads/audio.wav"
      if not os.path.exists(audio_path):
          print("\n🔍 Attempting to download YouTube audio...")
          try:
              audio_path, video_title = download_youtube_audio(url)
              print(f"✓ Downloaded: {video_title}")
          except Exception as e:
              print(f"\n❌ All download methods failed: {str(e)}")
              return False




      # Process audio into chunks
      print("\n✂️ Preparing audio chunks...")
      chunks = chunk_audio(audio_path, chunk_duration=30, overlap=5)
      print(f"Created {len(chunks)} chunks for processing")




      # Initialize Whisper
      print("\n🔊 Initializing Whisper transcription...")
      whisper = WhisperTranscriber()




      # Transcribe chunks
      print("\n🔄 Starting transcription (this may take several minutes)...")
      start_time = time.time()
      transcriptions = whisper.transcribe_chunks(chunks)
      transcribe_time = time.time() - start_time




      # Save results
      save_transcription(transcriptions)




      # Print summary
      total_text = " ".join(t['text'] for t in transcriptions)
      word_count = len(total_text.split())
    
      print("\n📊 Transcription Summary:")
      print(f"- Total time: {transcribe_time:.2f} seconds")
      print(f"- Word count: {word_count}")
      print(f"- Processing speed: {word_count/transcribe_time:.2f} words/sec")
    
      return True




  except Exception as e:
      print(f"\n❌ Transcription failed: {str(e)}")
      return False




def start_qa_session(question: str = ""):
  """Interactive Q&A session about the video content"""
  try:
      print("\n🔍 Loading Q&A system...")
      qa_agent = VideoQAAgent()
    
      print("\n💬 Q&A Session Started")
      print("---------------------")
      print("Ask questions about the video content. Examples:")
      print("- Summarize the key points")
      print("- What was said about [topic]?")
      print("- List the main ideas")
      print("Type 'exit' or 'quit' to end\n")
    
      if question:
          answer = qa_agent.ask_question(question)
          return answer
      else:
          return "Please provide a valid question to the Q&A system."
    
  except Exception as e:
      print(f"\n❌ Failed to start Q&A: {str(e)}")
      return "Q&A session failed."




if __name__ == "__main__":
  # This part will be removed since Gradio will handle URL input
  pass
