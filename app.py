import streamlit as st
from pdfminer.high_level import extract_text
from transformers import pipeline

# Page setup
st.set_page_config(page_title="EZ Works - Document Chat", layout="centered")
st.title("💬 Chat with Your Document – EZ Works AI Assistant")

# Session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# File upload
uploaded_file = st.file_uploader("📁 Upload a PDF or TXT document", type=["pdf", "txt"])
text = ""

if uploaded_file:
    # Extract and clean document text
    if uploaded_file.type == "application/pdf":
        text = extract_text(uploaded_file)
    else:
        text = uploaded_file.read().decode("utf-8")
    text = text.replace('\n', ' ').strip()
    context = text  # Use full text to avoid cutting off education section

    # Auto summary
    st.subheader("📌 Document Summary")
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    summary = summarizer(context[:1000], max_length=150, min_length=50, do_sample=False)[0]['summary_text']
    st.success(summary)

    # Examples
    st.markdown("🧠 **Try asking:**")
    st.markdown("- What are the skills mentioned?")
    st.markdown("- Mention the internships")
    st.markdown("- List the education qualifications")
    st.markdown("- What are the projects included?")

    # QA pipeline
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

    # Fallback lookup for education if QA fails
    def fallback_education_lookup():
        edu_keywords = ["B.Tech", "Bachelor", "M.Tech", "Master", "Graduated", "University", "College", "Degree"]
        matches = [sent for sent in text.split('.') if any(k in sent for k in edu_keywords)]
        return matches[0].strip() if matches else None

    def ask_document_qa(question):
        result = qa_pipeline(question=question, context=context)
        answer = result['answer'].strip()
        low_confidence = result['score'] < 0.3 or answer.lower() in ["", "n/a", "unknown", "no answer"]
        if low_confidence:
            # If vague question is about education, use fallback
            if "education" in question.lower() or "qualification" in question.lower() or "degree" in question.lower():
                fallback = fallback_education_lookup()
                return fallback if fallback else "🙇 Sorry, I couldn’t find anything related to that in the uploaded document."
            return "🙇 Sorry, I couldn’t find anything related to that in the uploaded document."
        return answer

    # Display past chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask something from the document...")

    if user_input:
        # User message
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Answer from QA or fallback
        answer = ask_document_qa(user_input)
        st.chat_message("assistant").markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("👆 Please upload a document to begin chatting.")







