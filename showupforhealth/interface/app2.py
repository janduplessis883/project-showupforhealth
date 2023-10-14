import streamlit as st
import pandas as pd
import io

def main():
    st.title("CSV Upload Example with Progress Bar")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        # To show progress bar
        progress_bar = st.progress(0)
        
        # Create a buffer to hold chunks of the uploaded file
        buffer = io.BytesIO()
        chunk_size = 8  # 8KB chunks
        total_size = 0

        # Read and buffer the file in chunks
        for chunk in iter(lambda: uploaded_file.read(chunk_size), b""):
            total_size += len(chunk)
            buffer.write(chunk)
            # Update the progress bar based on the chunks read
            progress = total_size / uploaded_file.size
            progress_bar.progress(progress)

        # Rewind the buffer to the beginning and read into a DataFrame
        buffer.seek(0)
        df = pd.read_csv(buffer)
        
        # Display DataFrame
        st.write(df)

if __name__ == "__main__":
    main()