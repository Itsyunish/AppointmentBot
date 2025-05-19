import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from datetime import datetime, timedelta
import random
from dotenv import load_dotenv

# Load environment variables
from dotenv import load_dotenv

load_dotenv()


api_key = os.getenv("GOOGLE_API_KEY")  
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key,
    temperature=0.7,
    client_options={"api_endpoint": "https://generativelanguage.googleapis.com"},
    transport="rest"
)


# Configure Streamlit
st.set_page_config(page_title="AI Assistant with Appointment Booking", layout="wide")
st.title("🤖 AI Assistant with Appointment Booking")

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'booking_stage' not in st.session_state:
    st.session_state.booking_stage = None
if 'appointment_details' not in st.session_state:
    st.session_state.appointment_details = {
        'date': None,
        'time': None,
        'first_name': None,
        'last_name': None,
        'email': None,
        'phone': None
    }

# Sample appointment database (in production use a real database)
appointment_db = {}

# Function to check appointment availability
def check_availability(date, time):
    # Convert to datetime object
    requested_datetime = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
    
    # Check if time is during working hours (9am-5pm)
    if requested_datetime.hour < 9 or requested_datetime.hour >= 17:
        return False, "Outside working hours (9am-5pm)"
    
    # Check if time is in 30-minute increments
    if requested_datetime.minute not in [0, 30]:
        return False, "Appointments are only available at :00 or :30"
    
    # Check if already booked (in a real app, query your database)
    for appt in appointment_db.values():
        appt_datetime = datetime.strptime(f"{appt['date']} {appt['time']}", "%Y-%m-%d %H:%M")
        if appt_datetime == requested_datetime:
            return False, "Time already booked"
    
    return True, "Available"

# Function to suggest alternative times
def suggest_alternatives(date, time):
    original_datetime = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
    suggestions = []
    
    # Check same day alternatives
    for delta in [-30, 30, -60, 60, -90, 90]:  # minutes
        new_time = original_datetime + timedelta(minutes=delta)
        if 9 <= new_time.hour < 17:  # Within working hours
            available, _ = check_availability(new_time.strftime("%Y-%m-%d"), new_time.strftime("%H:%M"))
            if available:
                suggestions.append(new_time.strftime("%H:%M"))
    
    # If no same-day alternatives, check next 3 days
    if not suggestions:
        for day_delta in [1, 2, 3]:
            new_date = original_datetime + timedelta(days=day_delta)
            for hour in [9, 10, 11, 12, 13, 14, 15, 16]:  # Check each hour
                for minute in [0, 30]:
                    new_time = new_date.replace(hour=hour, minute=minute)
                    available, _ = check_availability(new_time.strftime("%Y-%m-%d"), new_time.strftime("%H:%M"))
                    if available:
                        suggestions.append(new_time.strftime("%Y-%m-%d %H:%M"))
    
    return suggestions[:5]  # Return up to 5 suggestions

# Function to process uploaded documents
def process_documents(uploaded_files):
    documents = []
    for file in uploaded_files:
        file_path = f"./temp/{file.name}"
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        
        if file.name.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
            
        # Add other file type handlers here
        
        os.remove(file_path)
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    
    # Create embeddings and vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    # Create conversation chain
    st.session_state.conversation = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

# Function to handle document Q&A
def handle_document_query(query):
    if st.session_state.conversation:
        result = st.session_state.conversation({"query": query})
        st.session_state.chat_history.append(("user", query))
        st.session_state.chat_history.append(("assistant", result["result"]))
        return result["result"]
    return "Please upload documents first."

# Function to handle appointment booking
def handle_appointment_booking(user_input):
    response = ""
    
    if st.session_state.booking_stage == None:
        st.session_state.booking_stage = "date"
        return "I can help you book an appointment. What date would you like to book for? (YYYY-MM-DD)"
    
    elif st.session_state.booking_stage == "date":
        try:
            datetime.strptime(user_input, "%Y-%m-%d")
            st.session_state.appointment_details['date'] = user_input
            st.session_state.booking_stage = "time"
            return "What time would you like to book? (HH:MM)"
        except ValueError:
            return "Please enter a valid date in YYYY-MM-DD format."
    
    elif st.session_state.booking_stage == "time":
        try:
            datetime.strptime(user_input, "%H:%M")
            st.session_state.appointment_details['time'] = user_input
            
            # Check availability
            available, reason = check_availability(
                st.session_state.appointment_details['date'],
                st.session_state.appointment_details['time']
            )
            
            if available:
                st.session_state.booking_stage = "first_name"
                return "Great! That time is available. What's your first name?"
            else:
                alternatives = suggest_alternatives(
                    st.session_state.appointment_details['date'],
                    st.session_state.appointment_details['time']
                )
                if alternatives:
                    response = f"Sorry, that time is not available ({reason}). Here are some alternatives:\n"
                    for alt in alternatives:
                        response += f"- {alt}\n"
                    response += "Please select one or suggest another time."
                else:
                    response = "Sorry, no appointments available around that time. Please try a different date."
                return response
        except ValueError:
            return "Please enter a valid time in HH:MM format."
    
    elif st.session_state.booking_stage == "first_name":
        st.session_state.appointment_details['first_name'] = user_input
        st.session_state.booking_stage = "last_name"
        return "What's your last name?"
    
    elif st.session_state.booking_stage == "last_name":
        st.session_state.appointment_details['last_name'] = user_input
        st.session_state.booking_stage = "email"
        return "What's your email address?"
    
    elif st.session_state.booking_stage == "email":
        if "@" in user_input and "." in user_input:
            st.session_state.appointment_details['email'] = user_input
            st.session_state.booking_stage = "phone"
            return "Finally, what's your phone number?"
        else:
            return "Please enter a valid email address."
    
    elif st.session_state.booking_stage == "phone":
        if len(user_input) >= 7 and user_input.isdigit():
            st.session_state.appointment_details['phone'] = user_input
            
            # In a real app, save to database
            appointment_id = random.randint(1000, 9999)
            appointment_db[appointment_id] = st.session_state.appointment_details.copy()
            
            # Reset booking
            st.session_state.booking_stage = None
            details = st.session_state.appointment_details
            st.session_state.appointment_details = {k: None for k in details}
            
            return (f"Appointment booked successfully! 🎉\n"
                    f"Date: {details['date']}\n"
                    f"Time: {details['time']}\n"
                    f"Confirmation ID: {appointment_id}\n"
                    f"You'll receive a confirmation email at {details['email']}")
        else:
            return "Please enter a valid phone number (digits only)."

# Main chat interface
def main():
    # Create temp directory
    if not os.path.exists("./temp"):
        os.makedirs("./temp")
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("📂 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload documents for the AI to reference",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True
        )
        
        if uploaded_files and st.session_state.conversation is None:
            with st.spinner("Processing documents..."):
                process_documents(uploaded_files)
                st.success("Documents processed successfully!")
        
        st.markdown("---")
        st.markdown("**How to use:**")
        st.markdown("1. Upload documents for the AI to reference")
        st.markdown("2. Ask questions about the documents")
        st.markdown("3. Type 'book appointment' to start booking")
    
    # Chat container
    chat_container = st.container()
    
    # User input
    user_input = st.chat_input("Ask a question or type 'book appointment'...")
    
    if user_input:
        # Check if user wants to book appointment
        if "book appointment" in user_input.lower() and st.session_state.booking_stage is None:
            st.session_state.booking_stage = "date"
            st.session_state.chat_history.append(("user", user_input))
            st.session_state.chat_history.append(("assistant", 
                "I can help you book an appointment. What date would you like to book for? (YYYY-MM-DD)"))
        
        # Handle appointment booking flow
        elif st.session_state.booking_stage is not None:
            response = handle_appointment_booking(user_input)
            st.session_state.chat_history.append(("user", user_input))
            st.session_state.chat_history.append(("assistant", response))
        
        # Handle document Q&A
        else:
            response = handle_document_query(user_input)
    
    # Display chat history
    with chat_container:
        for role, message in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(message)

if __name__ == "__main__":
    main()