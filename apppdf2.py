import os
import time
from dotenv import load_dotenv
from groq import Groq
from langchain.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import streamlit as st
from fpdf import FPDF


# Function to load and process the documents from an uploaded PDF file
def get_docs(uploaded_file):
    start_time = time.time()
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    final_documents = text_splitter.split_documents(documents)
    st.sidebar.write('Documents Loaded')
    end_time = time.time()
    st.sidebar.write(f"Time taken to load documents: {end_time - start_time:.2f} seconds")
    os.remove("temp.pdf")  
    return final_documents

# Function to create vector store
def create_vector_store(docs):
    start_time = time.time()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"trust_remote_code": True})
    vectorstore = FAISS.from_documents(docs, embeddings)
    st.sidebar.write('DB is ready')
    end_time = time.time()
    st.sidebar.write(f"Time taken to create DB: {end_time - start_time:.2f} seconds")
    return vectorstore

# Function to interact with Groq AI
def chat_groq(messages):
    load_dotenv()
    client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
    response_content = ''
    stream = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        max_tokens=1024,
        temperature=1.3,
        stream=True,
    )

    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            response_content += chunk.choices[0].delta.content
    return response_content

# Function to summarize the chat history
def summarize_chat_history(chat_history):
    chat_history_text = " ".join([f"{chat['role']}: {chat['content']}" for chat in chat_history])
    prompt = f"Summarize the following chat history:\n\n{chat_history_text}"
    messages = [{'role': 'system', 'content': 'You are very good at summarizing the chat between User and Assistant'}]
    messages.append({'role': 'user', 'content': prompt})
    summary = chat_groq(messages)
    return summary

# Function to generate a PDF of chat history
from fpdf import FPDF

def generate_pdf(chat_history):
    pdf = FPDF()
    pdf.add_page()

    # Use the Arial font, which supports a wider range of Unicode characters
    pdf.add_font('Arial', '', 'arial.ttf', uni=True)
    pdf.set_font('Arial', '', 10)

    # Define margins
    left_margin = 15
    right_margin = 15
    top_margin = 15
    bottom_margin = 15

    # Set margins
    pdf.set_left_margin(left_margin)
    pdf.set_right_margin(right_margin)
    pdf.set_top_margin(top_margin)

    # Add a title
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Chat History', ln=True, align='C')
    pdf.ln(10)  # Add a line break

    pdf.set_font('Arial', '', 10)

    # Calculate maximum text width
    page_width = pdf.w - pdf.r_margin - pdf.l_margin

    # Iterate through chat history and add to PDF
    for chat in chat_history:
        role = chat['role'].capitalize()
        content = chat['content']
        text_to_add = f"{role}: {content}"

        # Check if text exceeds page width
        if pdf.get_string_width(text_to_add) > page_width:
            # Break text into chunks if necessary
            pdf.multi_cell(page_width, 10, text_to_add, border=0, align='L')
        else:
            # Add text to PDF with proper wrapping
            pdf.cell(0, 10, text_to_add, border=0, align='L')
        pdf.ln()  # Add a line break after each chat entry

    pdf_output = "chat_history.pdf"
    pdf.output(pdf_output)
    return pdf_output

# Main function to control the app
def main():
    st.set_page_config(page_title='Madhan_Doc_Bot')

    st.title("Madhan_Doc_Bot")
    with st.expander("Instructions to upload Text PDF/URL"):
        st.write("1. Pull up the side bar in top left corner.")
        st.write("2. If uploading a PDF, click 'Upload PDF', select your file, and wait for 'Documents Loaded' confirmation.")
        st.write("3. If entering a web URL, enter the URL, click 'Enter Web URL', and submit 'Process URL' and wait for 'Documents Loaded from URL' confirmation.")
        st.write("4. After loading documents, click 'Create Vector Store' to process.Documents can only be uploaded once per session")
        st.write("5. Enter a question in the text area and submit to interact with the AI chatbot.")
        st.write("6. Click on Generate Chat Summary to get the conversation of the Chat Session.")
        st.write("7. Click on Download Chat History to get a PDF of the chat history.")
        

    # Sidebar for document source selection
    st.sidebar.subheader("Choose document source:")
    option = st.sidebar.radio("", ("Upload PDF",))
    

    if "docs" not in st.session_state:
        st.session_state.docs = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    if "current_prompt" not in st.session_state:
        st.session_state.current_prompt = ""
    if "chat_summary" not in st.session_state:
        st.session_state.chat_summary = ""

    if option == "Upload PDF":
        uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
        if uploaded_file is not None:
            if st.session_state.docs is None:
                with st.spinner("Loading documents..."):
                    docs = get_docs(uploaded_file)
                st.session_state.docs = docs

    if st.session_state.docs is not None:
        if st.sidebar.button('Create Vector Store'):
            with st.spinner("Creating vector store..."):
                vectorstore = create_vector_store(st.session_state.docs)
            st.session_state.vectorstore = vectorstore

    if st.session_state.vectorstore is not None:
        def submit_with_doc():
            user_message = st.session_state.user_input
            if user_message:
                retriever = st.session_state.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
                context = retriever.invoke(user_message)
                prompt = f'''
                Answer the user's question based on the latest input provided in the chat history. Ignore
                previous inputs unless they are directly related to the latest question. Provide a generic
                answer if the answer to the user's question is not present in the context by mentioning it
                as general information.

                Context: {context}

                Chat History: {st.session_state.chat_history}

                Latest Question: {user_message}
                '''

                messages = [{'role': 'system', 'content': 'You are a very helpful assistant'}]
                messages.append({'role': 'user', 'content': prompt})

                try:
                    ai_response = chat_groq(messages)
                except Exception as e:
                    st.error(f"Error occurred during chat_groq execution: {str(e)}")
                    ai_response = "An error occurred while fetching response. Please try again."

                # Display the current output prompt
                st.session_state.current_prompt = ai_response

                # Update chat history
                st.session_state.chat_history.append({'role': 'user', 'content': user_message})
                st.session_state.chat_history.append({'role': 'assistant', 'content': ai_response})

                # Clear the input field
                st.session_state.user_input = ""

    def submit_without_doc():
        user_message = st.session_state.user_input
        if user_message:
            prompt = f'''
            Answer the user's question based on the latest input provided in the chat history. Ignore
            previous inputs unless they are directly related to the latest
            question. 
            
            Chat History: {st.session_state.chat_history}

            Latest Question: {user_message}
            '''

            messages = [{'role': 'system', 'content': 'You are a very helpful assistant'}]
            messages.append({'role': 'user', 'content': prompt})

            try:
                ai_response = chat_groq(messages)
            except Exception as e:
                st.error(f"Error occurred during chat_groq execution: {str(e)}")
                ai_response = "An error occurred while fetching response. Please try again."

            # Display the current output prompt
            st.session_state.current_prompt = ai_response

            # Update chat history
            st.session_state.chat_history.append({'role': 'user', 'content': user_message})
            st.session_state.chat_history.append({'role': 'assistant', 'content': ai_response})

            # Clear the input field
            st.session_state.user_input = ""

    # Display chat history and latest prompt
    st.write("")
    for chat in st.session_state.chat_history:
        st.write(f"{chat['role'].capitalize()}: {chat['content']}")

    # Display the current prompt if available
    if st.session_state.current_prompt:
        st.write("AI Response:")
        st.write(st.session_state.current_prompt)

    # Text input for user message
    st.text_area("Ask something", key="user_input", value=st.session_state.user_input, on_change=submit_with_doc if st.session_state.vectorstore else submit_without_doc)
    if st.session_state.vectorstore is not None:
        st.button('Submit', on_click=submit_with_doc)
    else:
        st.button('Submit', on_click=submit_without_doc)

    # Button to generate chat summary
    if st.button('Generate Chat Summary'):
        if st.session_state.chat_history:
            summary = summarize_chat_history(st.session_state.chat_history)
            st.session_state.chat_summary = summary
            st.write("Chat Summary:")
            st.write(summary)
        else:
            st.write("No chat history available to summarize.")



if __name__ == "__main__":
    main()
