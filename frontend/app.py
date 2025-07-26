import streamlit as st
import requests

st.title("Ask My Course - RAG Assistant")

# Input box for the user's question
question = st.text_input("Enter your question:")

# Button to submit the question
if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        # Send the question to the backend API
        try:
            # Change the URL below to match your FastAPI backend endpoint
            api_url = "http://localhost:8000/ask"  # Update if running backend elsewhere
            response = requests.post(api_url, json={"question": question})
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "[No answer returned]")
                st.success("Answer:")
                st.write(answer)
                # Optionally show context chunks
                if "context" in data:
                    with st.expander("Show supporting context"):
                        for i, chunk in enumerate(data["context"], 1):
                            st.markdown(f"**Chunk {i}:**\n{chunk}")
            else:
                st.error(f"Backend error: {response.status_code}")
        except Exception as e:
            st.error(f"Error contacting backend: {e}")

# (Optional) Instructions or footer
st.markdown("---")
st.markdown("_Powered by Streamlit, FastAPI, and your RAG pipeline!_")
