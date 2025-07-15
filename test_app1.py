
from flask import Flask, render_template, request, redirect, url_for
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS  # Or your preferred vector database
from langchain.chains import ConversationalRetrievalChain
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv() 


app = Flask(__name__)

# Global variables for storing video information
vector_store = None
chat_history = []
conversation_chain = None
youtube_url = None



# Initialize components
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")


def get_youtube_id(url):

    pattern = r'(?:https?://)?(?:www\.)?(?:youtu\.be/|youtube(?:-nocookie)?\.com/(?:embed/|v/|watch\?v=|shorts/|watch\?.+&v=))([\w-]{11})(?:.+)?'
    match = re.search(pattern, url)
    
    if match:
        return match.group(1)
    else:
        return None


def transcribe(video_id):
    try:
        # If you don’t care which language, this returns the “best” one
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])

        # Flatten it to plain text
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        print("/nTranscribed info- /n",transcript)
        
    except TranscriptsDisabled:
        print("No captions available for this video.")

    return transcript


def create_vectorstore(youtube_url):

    try:

        video_id = get_youtube_id(youtube_url)
        print('this is the youtube url',youtube_url)
        print('this is the video id',video_id)
        transcript = transcribe(video_id)
           
        # Split the transcript into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])

        
        # Create embeddings and store in FAISS vector store
        vector_store = FAISS.from_documents(chunks, embeddings)

        
        return vector_store

    except TranscriptsDisabled:
        print("No captions available for this video.")
        return None


@app.route("/", methods=["GET", "POST"])
def index():
    
    if request.method == "POST":
        youtube_url = request.form["youtube_link"]
        print('youtube url in index.html',youtube_url)
        if not youtube_url:
            return render_template("index.html", error="Please provide a YouTube URL.")
        
        return redirect(url_for('chat'))
    return render_template('index.html')


@app.route("/chat", methods=["GET", "POST"])
def chat():
    global chat_history, conversation_chain, vector_store, youtube_url

    youtube_url = request.form["youtube_link"]

    if not youtube_url:
        return redirect(url_for('index'))
    
    vector_store = create_vectorstore(youtube_url)

    if vector_store:
         retriever = vector_store.as_retriever()
         conversation_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=retriever,
                    return_source_documents=True  # Optional, to show sources
                )

    return redirect(url_for('gotochat'))


@app.route('/gotochat')
def gotochat():
    return render_template('chat.html')


@app.route('/ask_question', methods=["GET", "POST"])
def ask_question():
    global chat_history, conversation_chain
    
    if conversation_chain is None:
        return redirect(url_for('index')) # Redirect if no video has been processed
   
    user_question = request.form['question']

    result = conversation_chain({"question": user_question, "chat_history": chat_history})

    response = result["answer"]
    chat_history.append((user_question, response))

    return str(response)


if __name__ == '__main__':
    app.run(debug=True)