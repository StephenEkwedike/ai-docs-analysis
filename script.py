# Project Information:
# Python 3.10.6 was used to create this project.
# If you don't have the required libraries installed, you can install them using the commands in requirements.txt.
# Make sure to download and configure Tesseract OCR, as it's necessary for project functionality.
# You can find installation instructions for Tesseract OCR at https://github.com/tesseract-ocr/tesseract.
# Additionally, the project relies on Poppler utility. Follow these steps:
# 1. Download Poppler from https://github.com/oschwartz10612/poppler-windows/releases/tag/v23.08.0-0.
# 2. Add the downloaded Poppler executable files to your system's PATH for seamless integration.
# ------------------------------------------------------------------------------------------------------------------
# Create .env file and place your openai api key in the .env file
# Place the files u want to use for training in the data folder
import openai
import glob
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter
import shutil
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
import traceback
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from pdf2image import convert_from_path
import pytesseract
from dotenv import load_dotenv
from pydub import AudioSegment

# Load the environment variables from the .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

def split_audio(input_path, output_path):

    audio = AudioSegment.from_mp3(input_path)

    print("Length of original audio is", len(audio) / 1000, "seconds")
    chunk_length = 30 * 1000  
    chunks = [audio[i:i + chunk_length] for i in range(0, len(audio), chunk_length)]
    for i, chunk in enumerate(chunks):
        chunk.export(f"{output_path}/chunk-{i}.wav", format="wav")

    print(f"Successfully split the audio file into {len(chunks)} chunks.")


def process_audio_and_transcribe(input_audio_path, output_file_path, segments_dir):
    if input_audio_path.lower().endswith('.wav') or input_audio_path.lower().endswith('.mp3'):

        # Split audio into segments
        split_audio(input_audio_path, segments_dir)

        all_transcriptions = []
        for segment_filename in os.listdir(segments_dir):
            segment_path = os.path.join(segments_dir, segment_filename)
            with open(segment_path, "rb") as segment_file:
                transcript = openai.Audio.transcribe("whisper-1", file=segment_file)
                segment_transcript = transcript['text']
                all_transcriptions.append(segment_transcript)
                print(f"transcription complete for: {segment_filename}")

        # Combine transcriptions and save to a text file
        combined_transcription = "\n".join(all_transcriptions)
        with open(output_file_path, "w",encoding="utf-8") as output_file:
            output_file.write(combined_transcription)






def save_data(file_path, user_id):

    if file_path.lower().endswith('.pdf'):
        # Output directory for saving extracted text files
        try:
            print("Data extraction process started for pdf this will take a few mintutes Because we are performing Ocr On Every Page Your Pdf To Extract All Data ")
            output_dir = 'temp_pdf'
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)

            # Convert PDF to images and extract text
            pages = convert_from_path(file_path)
            all_text = []  # List to store extracted text from each page

            for i, page in enumerate(pages):
                image = page.convert('RGB')  # Convert to RGB mode
                text = pytesseract.image_to_string(image)
                all_text.append(text)

            # Save all extracted text to a single text file
            all_text_file_path = os.path.join(output_dir, f'{user_id}.txt')
            with open(all_text_file_path, 'w', encoding='utf-8') as text_file:
                text_file.write('\n\n'.join(all_text))

            print("All possible data extracted from pdf using OCR tesseract")


            persist_directory = f'trained_db/{user_id}/pdf_all_embeddings'

            loader = TextLoader(all_text_file_path, encoding='utf-8')
            documents = loader.load()
            text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=0)
            split_docs = text_splitter.split_documents(documents)
            embeddings = OpenAIEmbeddings()
            
            new_vectordb = FAISS.from_documents(split_docs, embeddings)
            try:
                old_vectordb = FAISS.load_local(persist_directory, embeddings)
                old_vectordb.merge_from(new_vectordb)
                old_vectordb.save_local(persist_directory)
                print("Embeddings were loaded For Pdf File")

            except:
                new_vectordb.save_local(persist_directory)
                print("New VectorStore is intialized For Pdf File")


        except Exception as e:
                # Log the traceback information to a file
                error_message = f"Error processing PDF: {str(e)}"
                traceback_str = traceback.format_exc()
                print(error_message)
                print(traceback_str)
                return {'answer': "Above error occured"}
        finally:
                shutil.rmtree(output_dir)

    elif file_path.lower().endswith('.docx'):
        
        persist_directory = f'trained_db/{user_id}/docx_all_embeddings'
        print("Data extraction proccess started for docx file")
        try:
              
            documents = UnstructuredWordDocumentLoader(file_path).load()
            text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=0)
            split_docs = text_splitter.split_documents(documents)
            embeddings = OpenAIEmbeddings()
            # Change to your directory
            new_vectordb = FAISS.from_documents(split_docs, embeddings)
            try:
                old_vectordb = FAISS.load_local(persist_directory, embeddings)
                old_vectordb.merge_from(new_vectordb)
                old_vectordb.save_local(persist_directory)
                print("Embeddings were loaded for docx file")
            except:
                new_vectordb.save_local(persist_directory)
                print("New VectorStore is intialized for docx file")
                

        except Exception as e:
                # Log the traceback information to a file
                error_message = f"Error processing DOCX: {str(e)}"
                traceback_str = traceback.format_exc()
                print(error_message)
                print(traceback_str)
                return {'answer': "Above error occured"}


    elif file_path.lower().endswith('.wav') or file_path.lower().endswith('.mp3'):
        # Create a directory for user-specific audio segments

        print('Data extraction Process started for audio file')
        try:
            segments_dir = f"segments_{user_id}"
            if os.path.exists(segments_dir):
                shutil.rmtree(segments_dir)
            os.makedirs(segments_dir)
            transcript_file_path = f"transcription_{user_id}.txt"
            if os.path.exists(transcript_file_path):
                os.remove(transcript_file_path)
            process_audio_and_transcribe(file_path, transcript_file_path, segments_dir)
            persist_directory = f'trained_db/{user_id}/audio_all_embeddings'
            loader = TextLoader(transcript_file_path, encoding="utf-8")
            documents = loader.load()
            text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=0)
            split_docs = text_splitter.split_documents(documents)

            # Local EMBEDDINGS
            embeddings = OpenAIEmbeddings()
            new_vectordb = FAISS.from_documents(split_docs, embeddings)
            try:
                old_vectordb = FAISS.load_local(persist_directory, embeddings)
                old_vectordb.merge_from(new_vectordb)
                old_vectordb.save_local(persist_directory)
                print("Embeddings were loaded for audio file")

            except:
                new_vectordb.save_local(persist_directory)
                print("New VectorStore is intialized for audio file")
                
            finally:
                os.remove(transcript_file_path)

        except Exception as e:
            error_message = f"Error processing wav or mp3 file transcript: {str(e)}"
            traceback_str = traceback.format_exc()
            print(error_message)
            print(traceback_str)
            return {'answer': "Above error occured"}
        finally:
            shutil.rmtree(segments_dir)



    else:
        raise ValueError("Supported audio formats are .wav, .mp3, docx, and pdf")



def delete(file, user_id ):
    if file=='audio':
        delete_directory =f'trained_db/{user_id}/audio_all_embeddings'
    elif file=='docx':
        delete_directory =f'trained_db/{user_id}/docx_all_embeddings'
    elif file=='pdf':
        delete_directory =f'trained_db/{user_id}/pdf_all_embeddings'

    if os.path.exists(delete_directory):
        shutil.rmtree(delete_directory)
        return {'answer': f"Embeddings for user {user_id} deleted successfully."}
     
    else:
        return {'answer': f"No embeddings found for user {user_id}."}
    

def chat(file, user_id, query):

    if file == 'audio':
        persist_directory =f'trained_db/{user_id}/audio_all_embeddings'
    elif file == 'docx':
        persist_directory =f'trained_db/{user_id}/docx_all_embeddings'
    elif file == 'pdf':
        persist_directory =f'trained_db/{user_id}/pdf_all_embeddings'

    # Run the search code and get the results
    try:
        embeddings = OpenAIEmbeddings()
        vectordb = FAISS.load_local(persist_directory, embeddings)
        # Build a QA chain
        qa_chain = RetrievalQA.from_chain_type(
        llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0),
        chain_type = "stuff",
        retriever= vectordb.as_retriever(),)
        ans=qa_chain.run(query)
        return {'answer': ans}

    except:
        return {'answer': f"No embeddings found for user {user_id} for the format u selected"}



def clear_screen():
    if os.name == 'posix':
        os.system('clear')
    else:
        os.system('cls')

def process_files_in_directory(directory):
    supported_formats = [".wav", ".pdf", ".docx", ".mp3"]
    files = []

    for format in supported_formats:
        files.extend(glob.glob(os.path.join(directory, '*' + format)))

    return files

if __name__ == "__main__":
    user_id = "222"
    try:
        while True:
            clear_screen()
            print("Select an option:")
            print("1. Save data")
            print("2. Delete previously saved data")
            print("3. Query data")
            print("4. Exit")

            choice = input("Enter choice (1/2/3/4): ")

            if choice == "1":
                num=1
                clear_screen()
                files = process_files_in_directory("data")  # Change "data" to your directory path
                for file_path in files:
                    clear_screen()
                    print(f'Data extraction process started for file # {num}')
                    message = save_data(file_path, user_id)
                    num=num+1
                    input("Press Enter to continue...")
            elif choice == "2":
                clear_screen()
                delete_format = input("Enter the format to delete (pdf/docx/audio): ")
                message = delete(delete_format, user_id)
                print(message)
                input("Press Enter to continue...")
            elif choice == "3":
                while True:
                    clear_screen()
                    query = input("Enter your query (or type 'back' to return to the main menu): ")
                    if query.lower() == 'back':
                        break
                    format_choice = input("Select a format to use (pdf/docx/audio): ")
                    message = chat(format_choice, user_id, query)
                    print(message)
                    input("Press Enter to continue...")
            elif choice == "4":
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please choose a valid option.")
                input("Press Enter to continue...")

    except ValueError as e:
        print(e)