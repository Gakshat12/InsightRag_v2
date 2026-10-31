"""Streamlit web interface for InsightRAG."""
import json
import requests
import streamlit as st

st.set_page_config(
    page_title="InsightRAG v3",
    page_icon="🔍",
    layout="wide"
)

st.title("InsightRAG v3")
st.caption("Hybrid RAG with BM25 + FAISS (RRF), CrossEncoder Reranking & Grounded Citations")

# Sidebar Configuration
with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", value="http://localhost:8000")
    top_k = st.slider("Top-K Chunks", min_value=1, max_value=20, value=5)
    rerank = st.checkbox("Enable CrossEncoder Reranker", value=True)
    stream_mode = st.checkbox("Streaming Mode (SSE)", value=True)

    st.markdown("---")
    st.subheader("System Health")
    try:
        health_resp = requests.get(f"{api_url}/health", timeout=3)
        if health_resp.status_code == 200:
            hdata = health_resp.json()
            status_color = "🟢" if hdata.get("index_loaded") else "🟡"
            st.write(f"{status_color} **Status:** {hdata.get('status')}")
            st.write(f"📚 **Corpus Chunks:** {hdata.get('corpus_size')}")
            st.write(f"🎯 **Reranker:** {'Enabled' if hdata.get('reranker_enabled') else 'Disabled'}")
        else:
            st.error(f"API Error {health_resp.status_code}")
    except Exception:
        st.warning("⚠️ API server unreachable at current URL.")

# Chat history initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "metadata" in msg:
            meta = msg["metadata"]
            if "confidence" in meta:
                conf = meta["confidence"]
                st.info(f"**Confidence Score:** `{conf.get('score', 0):.2%}` ({conf.get('method')}) | Grounded: {conf.get('grounded')}")
            if "citations" in meta and meta["citations"]:
                with st.expander(f"📚 Source Citations ({len(meta['citations'])})"):
                    for c in meta["citations"]:
                        st.markdown(f"**[{c['id']}] {c['source']} (Page {c['page']})** - Score: `{c['score']}`")
                        st.caption(f"> {c['snippet']}")

# Chat prompt input
user_query = st.chat_input("Ask a question about your documents...")

if user_query:
    # Append user question
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    payload = {
        "query": user_query,
        "top_k": top_k,
        "rerank": rerank,
        "stream": stream_mode,
    }

    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        full_text = ""
        meta_payload = {}

        if stream_mode:
            try:
                with requests.post(f"{api_url}/ask", json=payload, stream=True, timeout=60) as resp:
                    if resp.status_code != 200:
                        st.error(f"Error {resp.status_code}: {resp.text}")
                    else:
                        for line in resp.iter_lines():
                            if line:
                                line_str = line.decode("utf-8")
                                if line_str.startswith("data: "):
                                    data_body = line_str[6:].strip()
                                    if data_body == "[DONE]":
                                        break
                                    try:
                                        chunk_json = json.loads(data_body)
                                        if chunk_json.get("type") == "token":
                                            full_text += chunk_json.get("content", "")
                                            answer_placeholder.markdown(full_text + "▌")
                                        elif chunk_json.get("type") == "final":
                                            meta_payload = chunk_json
                                    except json.JSONDecodeError:
                                        continue
                        answer_placeholder.markdown(full_text)
            except Exception as e:
                st.error(f"Failed to communicate with API: {e}")
        else:
            try:
                resp = requests.post(f"{api_url}/ask", json=payload, timeout=60)
                if resp.status_code == 200:
                    meta_payload = resp.json()
                    full_text = meta_payload.get("answer", "")
                    answer_placeholder.markdown(full_text)
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

        # Display metadata (citations, confidence, latency)
        if meta_payload:
            conf = meta_payload.get("confidence", {})
            citations = meta_payload.get("citations", [])
            latencies = meta_payload.get("stage_latency_ms", {})
            tokens = meta_payload.get("token_usage", {})
            cost = meta_payload.get("estimated_cost_usd", 0.0)

            # Confidence Badge
            score_val = conf.get("score", 0.0)
            st.metric(
                label="Confidence Score",
                value=f"{score_val:.1%}",
                delta=f"Top: {conf.get('top_chunk_score')} | Avg: {conf.get('avg_chunk_score')}"
            )

            # Citations list
            if citations:
                with st.expander(f"📚 Source Citations ({len(citations)})", expanded=True):
                    for c in citations:
                        st.markdown(f"**[{c['id']}] {c['source']} (Page {c['page']})** - Score: `{c['score']}`")
                        st.caption(f"> {c['snippet']}")

            # Metrics
            with st.expander("⚡ Latency & Cost Breakdown"):
                col1, col2, col3 = st.columns(3)
                col1.write(f"**Total Latency:** {meta_payload.get('total_latency_ms')} ms")
                for stage, ms in latencies.items():
                    col1.caption(f"- {stage}: {ms} ms")
                col2.write(f"**Tokens:** {tokens.get('total_tokens', 0)}")
                col2.caption(f"- Prompt: {tokens.get('prompt_tokens', 0)}")
                col2.caption(f"- Completion: {tokens.get('completion_tokens', 0)}")
                col3.write(f"**Estimated Cost:** ${cost:.6f}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_text,
                "metadata": meta_payload,
            })

