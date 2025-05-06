import gradio as gr
import sys
import os


# Add the parent directory of 'main.py' to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Now import main from the correct path


from main import transcribe_audio, start_qa_session


def process_video(video_url):
   # Run transcription
   transcription_success = transcribe_audio(video_url)
  
   if transcription_success:
       return "Transcription successful. You can now start asking questions."
   else:
       return "Transcription failed. Please check the video link or try again."


def answer_question(question):
   # Call the Q&A session to get the answer
   answer = start_qa_session(question)
   return answer


# Gradio Interface
with gr.Blocks() as app:
   gr.Markdown("### YouTube Video Transcription and Q&A System")
  
   # Video URL Input
   video_url_input = gr.Textbox(label="Enter YouTube Video URL", placeholder="Enter a valid YouTube video link...")
  
   # Button to start transcription and Q&A
   start_button = gr.Button("Start Transcription and Q&A")
  
   # Output area to show if transcription was successful
   output_text = gr.Textbox(label="Status", interactive=False)
  
   # Question Input
   question_input = gr.Textbox(label="Ask a question about the video", placeholder="Enter your question here...")
  
   # Answer Output
   answer_output = gr.Textbox(label="Answer", interactive=False)
  
   # Define button actions for transcription
   start_button.click(process_video, inputs=[video_url_input], outputs=[output_text])
  
   # Define button actions for submitting the question
   submit_button = gr.Button("Send Question")
   submit_button.click(answer_question, inputs=[question_input], outputs=[answer_output])


# Launch the Gradio app
app.launch(share=True)