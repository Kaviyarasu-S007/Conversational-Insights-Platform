# Conversational-Insights-Platform

Overview
The Conversational Insights Platform is a comprehensive tool designed to process video and audio files, extracting valuable information such as transcriptions, topics, sentiments, and actionable insights. Developed using Flask, NLP, Deep Learning, and ASR technologies, this platform aims to provide in-depth analysis and understanding of audio-visual content.

Table of Contents
Introduction
Tech Stack
Tasks and Implementation
Task 1 - Data Ingestion
Task 2 - Transcription
Task 3 - Topic Extraction
Task 4 - Metadata Extraction
Task 5 - Sentiment Analysis
Task 6 - Insight Generation
Task 7 - User Interface
Task 8 - LLM Integration
Task 9 - Version Control
Task 10 - Documentation
Usage
Future Work
License
Introduction
The Conversational Insights Platform processes and analyzes video and audio files to provide transcriptions, identify topics, perform sentiment analysis, and generate actionable insights. This project leverages various technologies to ensure robust and accurate processing.

Tech Stack
Backend: Flask (Python)
Frontend: HTML, CSS
Libraries: NLP, ASR, and sentiment analysis libraries
Visualization: Word Cloud, Matplotlib, Seaborn
Tasks and Implementation
Task 1 - Data Ingestion
Description: Supports uploading video and audio files (MP4, MP3, WAV).
Implementation:
Built a file upload feature using Flask.
Processed video formats with MoviePy and audio with SpeechRecognition.
Ensured secure file handling using werkzeug.utils.secure_filename.
Code Reference: Flask app.py - /upload route

![Screenshot 2024-09-16 053506](https://github.com/user-attachments/assets/c87648c9-e95a-4a3f-9170-974344ab2f62)



Task 2 - Transcription
Description: Transcribes audio/video files into text using ASR.
Implementation:
Extracted audio from video files using MoviePy.
Converted audio to text using SpeechRecognition.
Handled noisy audio with librosa.
Speaker Diarization: Identified speakers using Hugging Face models.
Code Reference: Transcription functionality (app.py)

![Screenshot 2024-09-16 053543](https://github.com/user-attachments/assets/0f374d8c-1601-4260-bf18-e351faaa2416)


Task 3 - Topic Extraction
Description: Identified key topics from transcribed text.
Implementation:
Applied LDA (Latent Dirichlet Allocation) to extract topics.
Visualized topics using word clouds and coherence scores.
Enhanced understanding with topic distribution graphs.
Code Reference: Topic extraction pipeline (topic_extraction.py)

![Screenshot 2024-09-16 083136](https://github.com/user-attachments/assets/976216c6-af84-45da-a482-a83a888f7c5c)


Task 4 - Metadata Extraction
Description: Extracted metadata such as speaker count, duration, and language.
Implementation:
Used Hugging Face model for speaker diarization.
Extracted conversation metadata: word count, duration, and keyword frequency.
Visualized metadata with charts and keyword frequency graphs.
Code Reference: Metadata extraction (metadata.py)


Task 5 - Sentiment Analysis
Description: Classified speaker sentiments (positive, negative, neutral).
Implementation:
Started with Hugging Face sentiment models, later switched to TextBlob for efficiency.
Visualized speaker sentiments with bar charts and sentiment distribution graphs.
Code Reference: Sentiment analysis using TextBlob (sentiment_analysis.py)


Task 6 - Insight Generation
Description: Generated actionable insights from transcribed conversations.
Key Insights:
Customer Satisfaction Levels
Common Complaints
Market Trends
Retention Metrics
Code Reference: Insight generation process (insight_generation.py)


Task 7 - User Interface
Description: Built a user-friendly dashboard for uploading files and viewing processed results.
Implementation:
Flask-based UI for displaying transcription, topic modeling, and sentiment analysis.
Separate pages for each output category.
UI Preview: Screenshot of dashboard features.
Code Reference: HTML templates and Flask routes (app.py, templates/)


Task 8 - LLM Integration
Description: Integrated a Large Language Model for advanced query responses and summary generation.
Implementation:
Used Hugging Face open-source LLM for contextual understanding.
Provides summaries and answers specific queries.
Code Reference: LLM integration (bot.py)


Task 9 - Version Control
Description: Used Git for version control.
Implementation:
Tracked code updates, maintained separate branches for each feature.
README provided instructions for setup and usage.
Code Reference: GitHub repository overview (screenshots)


Task 10 - Documentation
Description: Detailed documentation on the project pipeline.
Reporting:
Accuracy of transcription
Topic extraction (LDA coherence score)
Challenges faced during sentiment analysis

Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/your-repository.git
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the Flask application:

bash
Copy code
python app.py
