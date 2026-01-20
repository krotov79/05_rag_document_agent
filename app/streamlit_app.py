import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG Document Agent", page_icon="📄", layout="centered")

st.title("📄 RAG Document Agent")
st.caption("Ask questions over your small document corpus (FAISS + embeddings + Gemini).")

question = st.text_area("Your question", height=120, placeholder="What do HTTP 5xx status codes mean?")
k = st.slider("Top-k passages", 1, 10, 5)
min_score = st.slider("Min similarity score", 0.0, 1.0, 0.35)

if st.button("Ask", type="primary"):
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        with st.spinner("Retrieving + generating..."):
            r = requests.post(f"{API_BASE}/ask", json={"question": question, "k": k, "min_score": min_score}, timeout=60)
        if r.status_code != 200:
            st.error(f"API error: {r.status_code}\n{r.text}")
        else:
            data = r.json()
            st.subheader("Answer")
            st.write(data.get("answer", ""))

            passages = data.get("passages", [])
            if passages:
                st.subheader("Sources")
                for p in passages:
                    st.markdown(f'**[{p["citation"]}]** score={p["score"]:.3f}')
                    st.write(p["text"])
                    st.divider()
