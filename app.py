import streamlit as st
from logic import * 
import os 

def main():
    st.set_page_config(layout="wide")
    
    # Initialize status
    status = "Empty"
    
    # Initialize session state for selected model
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "gpt-4o"  # Default model
        # Initial LLM load
        LLMCache.get_llm(st.session_state.selected_model)
    
    # Sidebar
    with st.sidebar:
        # Model Selection
        st.title("Model Selection")
        st.session_state.selected_model = st.selectbox(
            "Choose a model", 
            [
                "gpt-4o", 
                "gpt o1", 
                "Claude Sonnet", 
                "gemini 1.5 pro"
            ],
            index=0,  # Default to first model
            key="model_selectbox",
            on_change=lambda: LLMCache.get_llm(st.session_state.selected_model)
        )
        
        # File Uploader
        st.title("Document Upload")
        uploaded_file = st.file_uploader(
            "Upload a file", 
            type=["txt", "pdf", "csv"],
        )
        
        file_path = None
        if uploaded_file is not None:
            # Create a directory to store uploads
            os.makedirs('uploads', exist_ok=True)
            # Save the file
            file_path = os.path.join('uploads', uploaded_file.name)
            #Stage 1: Loading
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            status = "Ready to Index"

        # Status Indicator
        st.title("Indexing Status")
        if uploaded_file is None:
            status = "Empty"
        else:
            # Add your indexing logic here
            if st.button("Index Document"):
                if index_document(file_path):
                    status = "Indexed"

        st.write(f"Current Status: **{status}**")

    # Chatbot Main Area
    st.title("Compare LLMs Chatbot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            # Generate response 
            response = ask_llm(prompt)
            
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(f"Model: {st.session_state.selected_model}")
                st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        except Exception as e:
            st.error(f"Error generating response: {e}")

if __name__ == "__main__":
    main()