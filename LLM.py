import os
import librosa
import ffmpeg
from moviepy.editor import VideoFileClip
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import whisper
import torch
from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer, util

# Initialize the Google Palm LLM
llm = GooglePalm(google_api_key="AIzaSyCWACMY_GG0FhNlBXQ5B1eOgrthbDf9Rjw", temperature=0.1)

# Initialize the embeddings using the Hugging Face model
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create a FAISS instance for the vector database (assuming the documents are preloaded)
vectordb = FAISS()  # Placeholder. You'll load the FAISS vector store with your documents.

# Create a retriever for querying the vector database
retriever = vectordb.as_retriever(score_threshold=0.7)

# Define the prompt template
prompt_template = """Given the following context and a question, generate an answer based on this context only.
In the answer, try to provide as much text as possible from the "response" section in the source document context without making many changes.
If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

CONTEXT: {context}

QUESTION: {question}"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

# Initialize the RetrievalQA chain
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    input_key="query",
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

# Load a pre-trained sentence transformer model for semantic similarity checking
similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Function to handle audio format conversion to WAV
def convert_to_wav(input_file, output_file="output_audio.wav"):
    ext = os.path.splitext(input_file)[1].lower()
    if ext == '.mp3' or ext == '.wav':
        print(f"Converting {input_file} to WAV format...")
        ffmpeg.input(input_file).output(output_file).run()
    elif ext in ['.mp4', '.mkv']:
        print(f"Extracting audio from video file {input_file}...")
        video_clip = VideoFileClip(input_file)
        video_clip.audio.write_audiofile(output_file)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    return output_file

# Function to load and transcribe audio using Whisper
def transcribe_audio_whisper(audio_file):
    print(f"Transcribing {audio_file} using Whisper...")
    model = whisper.load_model("base")  # Choose a model size as needed
    result = model.transcribe(audio_file)
    return result['text']

# Function to transcribe audio using Wav2Vec2
def transcribe_audio_wav2vec2(audio_file):
    print(f"Transcribing {audio_file} using Wav2Vec2...")
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
    audio_input, sample_rate = librosa.load(audio_file, sr=16000)
    input_values = tokenizer(audio_input, return_tensors="pt", padding="longest").input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.decode(predicted_ids[0])
    return transcription

# Function to pass transcribed text as a query to the LLM
def ask(query):
    if not query:
        return {"error": "No query provided"}
    result = chain({"query": query})
    context = result['source_documents'][0].page_content
    query_embedding = similarity_model.encode(query, convert_to_tensor=True)
    context_embedding = similarity_model.encode(context, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(query_embedding, context_embedding)
    print(f"Similarity Score: {similarity_score}")
    similarity_threshold = 0.7
    if similarity_score < similarity_threshold:
        return {"response": "I don't know."}
    return {"response": result['result']}

# Main function to handle ingestion, format conversion, transcription, and querying
def process_file(input_file, use_whisper=True):
    try:
        # Convert input file to WAV
        wav_file = convert_to_wav(input_file)

        # Perform transcription using Whisper or Wav2Vec2
        if use_whisper:
            transcription = transcribe_audio_whisper(wav_file)
        else:
            transcription = transcribe_audio_wav2vec2(wav_file)

        # Query the LLM using the transcription
        response = ask(transcription)
        print("LLM Response:", response['response'])

    except Exception as e:
        print(f"Error processing file: {e}")

# Example usage
if _name_ == "_main_":
    input_file = "sample_video.mp4"  # Example input file
    process_file(input_file, use_whisper=True)  # Change to False for Wav2Vec2 transcription