import os
import pandas as pd

from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

VECTOR_DB_PATH = "vector_db"

def create_rag_system(csv_path, memory):
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
        
        # Updated embeddings initialization
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        if os.path.exists(os.path.join(VECTOR_DB_PATH, "index.faiss")):
            print("Loading existing FAISS vectorstore...")
            vectorstore = FAISS.load_local(
                VECTOR_DB_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            print("⚙️ Creating new FAISS vectorstore...")
            structured_docs = []
            unstructured_docs = []

            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)

                for idx, row in df.iterrows():
                    content = f"""PG Record {idx + 1}:
                            - ID: {row.get('ID', 'N/A')}
                            - Name: {row.get('Name', 'N/A')}
                            - Age: {row.get('Age', 'N/A')}
                            - Gender: {row.get('Gender', 'N/A')}
                            - Profession: {row.get('Profession', 'N/A')}
                            - Profession Category: {row.get('Profession_Category', 'N/A')}
                            - Company: {row.get('Company', 'N/A')}
                            - Working Mode: {row.get('Working_Mode', 'N/A')}
                            - City: {row.get('City', 'N/A')}
                            - PG Name: {row.get('PG_Name', 'N/A')}
                            - PG Type: {row.get('PG_Type', 'N/A')}
                            - Languages Spoken: {row.get('Languages_Spoken', 'N/A')}
                            - Rent (INR): {row.get('Rent (INR)', 'N/A')}
                            - Stay Duration (Months): {row.get('Stay_Duration_Months', 'N/A')}
                            - Join Date: {row.get('Join_Date', 'N/A')}
                            - PG Rating: {row.get('PG_Rating', 'N/A')}
                            - Amenities Used: {row.get('Amenities_Used', 'N/A')}"""

                    structured_docs.append(Document(
                        page_content=content.strip(),
                        metadata={"source": "csv_data", "row_id": idx}
                    ))
                
                # Schema document generation
                schema_doc = f"""📄 **Dataset Schema Overview**

                This PG (Paying Guest) accommodation dataset contains **{len(df)} records** with the following columns:

                - ID: Unique identifier for each resident
                - Name: Resident's full name
                - Age: Age of the resident
                - Gender: Gender identity
                - Profession: Specific job title
                - Profession_Category: Broader category (e.g., IT, Finance)
                - Company: Organization the resident is employed at
                - Working_Mode: WFH, Office, or Hybrid
                - City: PG location
                - PG_Name: Name of the PG
                - PG_Type: Type (Boys, Girls, Co-living)
                - Languages_Spoken: Languages known by the resident
                - Rent (INR): Monthly rent in Indian Rupees
                - Stay_Duration_Months: Length of stay
                - Join_Date: Date resident joined the PG
                - PG_Rating: User rating (1–5 scale)
                - Amenities_Used: Facilities accessed

                ---

                📊 **Schema Summary**

                - **Unique Cities**: {', '.join(df['City'].dropna().unique()) if 'City' in df.columns else 'Unknown'}
                - **PG Types**: {', '.join(df['PG_Type'].dropna().unique()) if 'PG_Type' in df.columns else 'Unknown'}
                - **Working Modes**: {', '.join(df['Working_Mode'].dropna().unique()) if 'Working_Mode' in df.columns else 'Unknown'}
                - **Profession Categories**: {', '.join(df['Profession_Category'].dropna().unique()) if 'Profession_Category' in df.columns else 'Unknown'}
                - **Languages (sample)**: {', '.join(df['Languages_Spoken'].dropna().unique()[:10]) + ', ...' if 'Languages_Spoken' in df.columns else 'Unknown'}
                - **Rent Range**: ₹{int(df['Rent (INR)'].min())} to ₹{int(df['Rent (INR)'].max())} per month
                - **Average Stay Duration**: {round(df['Stay_Duration_Months'].mean(), 2)} months
                - **Average PG Rating**: {round(df['PG_Rating'].mean(), 2)} / 5

                ---
                💡 Use this metadata to ground your responses in the dataset structure.
                """
                unstructured_docs.append(Document(page_content=schema_doc.strip(), metadata={"source": "schema"}))

            # Domain contexts
            domain_contexts = [
                """PG Accommodation Guide:
                PG (Paying Guest) accommodations are shared living spaces popular in Indian cities. Key factors to consider:
                - Location: Proximity to workplace/transport
                - Rent: Budget-friendly options with inclusive bills
                - Amenities: WiFi, meals, laundry, AC, security
                - Type: Boys/Girls/Co-living options
                - Safety: 24/7 security, CCTV, safe neighborhood
                - Community: Language preferences, professional networking""",

                """Rent Analysis Insights:
                - Metropolitan cities (Mumbai, Delhi, Bangalore) have higher rents
                - IT professionals often prefer co-living spaces
                - Working mode affects location preferences (WFH vs Office)
                - PG ratings correlate with amenities and management quality
                - Stay duration indicates satisfaction levels""",

                """Professional Categories:
                - IT/Software: High demand for PGs near tech hubs
                - Healthcare: Preference for locations near hospitals
                - Finance: Usually located in business districts
                - Students: Budget-conscious, prefer basic amenities
                - Working professionals: Value convenience and networking"""
            ]
            for i, text in enumerate(domain_contexts):
                unstructured_docs.append(Document(page_content=text.strip(), metadata={"source": "domain", "id": i}))

            # Text splitting with updated parameters
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=100,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            split_unstructured = splitter.split_documents(unstructured_docs)

            all_docs = structured_docs + split_unstructured

            vectorstore = FAISS.from_documents(all_docs, embeddings)
            vectorstore.save_local(VECTOR_DB_PATH)

        # Updated prompt template
        custom_prompt = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template=""" You are a data assistant that ONLY answers using the provided PG dataset. Never use external knowledge or make assumptions.
                        **STRICT RULES:**
                        - Use ONLY information from the provided context
                        - If information is missing, respond: "This information is not available in the provided dataset."
                        - Never provide general advice or external knowledge
                        - Always specify data source (e.g., "Based on the dataset...")
                        - If user say Hi/hello always greet them

                        **Conversation History:**
                        {chat_history}

                        **Dataset Context:**
                        {context}

                        **Question:**
                        {question}

                        **Dataset Columns:**
                        ID, Name, Age, Gender, Profession, Profession_Category, Company, Working_Mode, City, PG_Name, PG_Type, Languages_Spoken, Rent (INR), Stay_Duration_Months, Join_Date, PG_Rating, Amenities_Used

                        **Response Format:**
                        - If data available: Provide specific values/statistics from dataset
                        - If data unavailable: "This information is not available in the provided dataset."
                        - Start answers with "Based on the dataset..." or "The data shows..."

                        Answer: """
        )

        # Updated conversational RAG chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 10}
            ),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": custom_prompt},
            return_source_documents=False
        )

        return qa_chain

    except Exception as e:
        print(f"Error setting up RAG system: {e}")
        return None


def query_rag(qa_chain, question, memory=None):
    """
    Query the RAG system with improved error handling
    """
    try:
        if not qa_chain:
            return "RAG system not available"
        
        result = qa_chain.invoke({"question": question})
        if question.strip().lower() in {"hi", "hello", "hey"}:
            return "Hello! How can I assist you with the PG dataset today?"
        return result["answer"]
    
    except Exception as e:
        return f"Error with RAG query: {str(e)}"