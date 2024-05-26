import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import pandas as pd 
import streamlit as st


from pathlib import Path
parent_dir = Path(__file__).parent.parent
db_path = str(parent_dir) + "/test_index"  

pd.set_option('display.max_columns', 4)

class ChromaViewer():
    def __init__():
        pass


    def view_collections(db_path):
        client = chromadb.PersistentClient(path=db_path)

        st.subheader("ChromaDB Collections")
        for collection in client.list_collections():
            data = collection.get(include=['embeddings', 'documents', 'metadatas'])

            # ids = data['ids']
            # embeddings = data["embeddings"]
            # metadata = data["metadatas"]
            # documents = data["documents"]

            df = pd.DataFrame.from_dict(data)
            st.markdown(f"{df.shape}")
            st.dataframe(df)

    

if __name__ == "__main__":
    try:
        print(f"Opening database: {db_path}")
        cv = ChromaViewer.view_collections
        cv(db_path)
    except:
        pass