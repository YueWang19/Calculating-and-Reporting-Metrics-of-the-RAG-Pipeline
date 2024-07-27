import streamlit as st
from ragchatbotv2 import get_answers

# Initialize session state for conversation history
if 'history' not in st.session_state:
    st.session_state.history = []

# Custom CSS for chat messages
st.markdown("""
    <style>
    .user-message {
        display: flex;
        align-items: center;
        background-color: #DCF8C6;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        width: fit-content;
    }
    .bot-message {
        display: flex;
        align-items: center;
        background-color: #F1F0F0;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        width: fit-content;
    }
    .user-image, .bot-image {
        width: 40px;
        height: 40px;
        margin-right: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Display emojis and welcome message
st.markdown("## 👰 🤵 Wedding music recommendation Chatbot")
st.markdown("Welcome! How can I assist you today?")

# # Input box for user query
# query = st.text_input("Enter your question:")

# Form for user query input and submit button
with st.form(key='query_form', clear_on_submit=True):
    query = st.text_input("Enter your question:")
    submit_button = st.form_submit_button(label='Ask')

# Process user query
if query:
    answer = get_answers(query)
    # Add user query and bot response to conversation history
    st.session_state.history.append({"message": query, "is_user": True})
    st.session_state.history.append({"message": answer, "is_user": False})

# Display conversation history in reverse order
for chat in reversed(st.session_state.history):
    if chat["is_user"]:
        st.markdown(f"""
            <div class="user-message">
                <div class="user-image">👰</div>
                <div>{chat["message"]}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="bot-message">
                <div class="bot-image">😊</div>
                <div>{chat["message"]}</div>
            </div>
            """, unsafe_allow_html=True)
        

