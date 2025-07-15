# Youtube Learning Assistant
![demo image 01](https://github.com/rajeshchalla10/YouTube_Learning_Assistant_RAG/blob/4c4a3716ba15e313bc001c80187fe6c7fc07b4b2/demo%20image/01%20demo%20youtube%20learning%20assistant.png)![demo image 02](https://github.com/rajeshchalla10/YouTube_Learning_Assistant_RAG/blob/63be04d19d3fdd5d88eeba385e905b84e90aab26/demo%20image/02%20demo%20youtube%20learning%20assistant.png)
<img src="https://github.com/rajeshchalla10/YouTube_Learning_Assistant_RAG/blob/4c4a3716ba15e313bc001c80187fe6c7fc07b4b2/demo%20image/01%20demo%20youtube%20learning%20assistant.png" width="425"/>
## Overview

This project presents a Youtube Learning Assistant built using Python, Flask, Langchain, and RAG (Retrieval-Augmented Generation) based of Large Language Model (LLM). This application allows users to ask questions and get information from any YouTube video link by leveraging the video transcript and RAG system. 

## Key Features
- YouTube Video Link Processing: Input any YouTube video link to initiate the learning process.
- Transcript Extraction: Extracts the transcript from the given YouTube video using YouTubeTranscriptApi.
- Text Chunking and Storage: Splits the extracted transcript into manageable chunks using RecursiveCharacterTextSplitter and stores them in a FAISS Vector Database.
- Q&A with RAG: Answers user questions based on the video's content using a Retrieval-Augmented Generation (RAG) approach, retrieving relevant information from the transcript chunks.
- We can use this system for Video Summarization (Optional).
- Flask Web Interface: Provides a simple web interface for users to interact with the assistant.

## Technologies Used
- Python: The core programming language for the application.
- Flask: A lightweight web framework used for building the application's interface.
- Langchain: A framework for developing applications powered by language models, used for transcript loading, chunking, and RAG implementation.
- Large Language Model (LLM): Powers the question-answering and potentially summarization capabilities. Can be integrated with open-source LLMs or services like GEMINI API.
- Retrieval-Augmented Generation (RAG): A technique that improves LLM performance by retrieving information from a knowledge base to augment the generation process.

## Getting Started

### Prerequisites
- Check the requirements.txt file to know the required libraries to install.
  
### Usage
- Open your web browser and navigate to http://localhost:5000 (or the port specified by the application).
- Enter a YouTube video URL into the provided input field on the web interface.
- Submit the URL to initiate the processing.
- Ask questions related to the video content in the chat interface. The assistant will retrieve relevant information and provide answers based on the video's transcript.

## Contributing
Contributions are welcome! If you have suggestions or improvements, feel free to fork the repository, make your changes, and submit a pull request.
Enter a YouTube video URL into the provided input field on the web interface.
