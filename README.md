Drive for screen recording on testing videos: https://drive.google.com/drive/folders/1V_plt49u-6tefiu4ay2QLHw0pREbMinm?usp=share_link

YouTube Video Transcription and Q&A System

Overview

This project provides an automatic transcription of YouTube videos and an interactive Q&A system that allows users to ask questions based on the transcribed content. The system integrates multiple AI tools, including Whisper for transcription, LangChain for routing queries, and Chroma for storing transcriptions in a vector database.

The system is designed to be cost-effective, with local Whisper deployment and the use of ChatGPT 3.5 for Q&A. This ensures both reduced processing costs and efficient performance.

Features

YouTube Video Downloading: Downloads audio from YouTube videos for transcription.
Local Transcription: Utilizes the Whisper model for transcribing audio to text locally, reducing costs.
Q&A System: Uses LangChain for handling queries and routing them to the transcription data for answer generation.
Vector Database: Stores transcriptions in Chroma for fast retrieval and efficient search.
Cost-Efficiency Focus: Optimized for local processing to minimize reliance on cloud-based services, lowering overall costs.
Architecture

The system architecture involves the following components:

YouTube Downloader: Downloads the audio from a YouTube video.
Whisper Transcriber: Transcribes the downloaded audio into text.
Text Chunking: The audio is chunked into smaller segments to improve transcription accuracy.
LangChain Agents: Route user queries and interact with the transcription data.
Vector Database (Chroma): Stores and retrieves transcription data for quick response generation.
Q&A Interface: A Gradio interface that allows users to input questions about the video content.
Key Technologies Used:
Whisper: For local transcription of audio.
LangChain: For agent-based handling of queries.
Chroma: For storing and retrieving transcriptions using vector embeddings.
Gradio: For the interactive user interface.
Requirements

Python 3.8+
Required packages (listed in requirements.txt)
To install dependencies, use:

pip install -r requirements.txt
Setup

Clone this repository:
git clone [https://github.com/yourusername/youtube-qa-chatbot.git]
Install the necessary dependencies:
pip install -r requirements.txt
Create a .env file and add your LangChain API key:
LANGCHAIN_API_KEY=your_langchain_api_key
Run the system:
python app.py
This will start the Gradio interface for YouTube video transcription and Q&A.

Usage

Input YouTube URL: Enter a valid YouTube video URL in the input box.
Start Transcription: Click "Start Transcription and Q&A" to begin transcribing the video.
Ask Questions: Once transcription is complete, ask any question related to the video, and the system will provide an answer based on the transcription.
Performance Evaluation

The system performance was evaluated by comparing different Whisper models, such as Whisper-Tiny and Whisper-Base, on processing time and transcription accuracy. The Whisper-Tiny model was chosen for its balance of speed and accuracy, making it ideal for local, cost-effective deployment.

Future Work

Cloud Support: Enhance the system to process large videos by integrating cloud-based resources.
Improved Q&A Accuracy: Train the Q&A system with a broader dataset to improve the quality and relevance of answers.
Scalability: Explore options to scale the system for batch processing of multiple videos.
