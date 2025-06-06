import streamlit as st
import requests
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import pandas as pd
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Google autocomplete function
def get_google_suggestions(keyword):
    url = f"https://suggestqueries.google.com/complete/search?client=firefox&q={keyword}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()[1]
    return []

# Intent tagging
def get_intent(suggestion):
    suggestion_lower = suggestion.lower()
    if any(word in suggestion_lower for word in ["how", "what", "why", "guide", "learn", "tutorial"]):
        return "Informational"
    elif any(word in suggestion_lower for word in ["buy", "price", "deal", "cheap", "best", "near me"]):
        return "Transactional"
    elif any(word in suggestion_lower for word in ["vs", "compare", "difference"]):
        return "Comparative"
    else:
        return "Other"

# Cluster labeling
def label_cluster(topics):
    words = " ".join(topics).lower().split()
    common = Counter(words).most_common(3)
    return " / ".join([w for w, _ in common])

# Word cloud generation
def show_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(text))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

# Clustering
def generate_clusters(seed):
    prompts = [f"{q} {seed}" for q in ["what is", "how to", "why", "can", "does", "vs", "near me", "examples of"]]
    suggestions = []
    for prompt in prompts:
        suggestions += get_google_suggestions(prompt)

    # Clean and deduplicate
    def clean_text(text):
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    unique_suggestions = list({clean_text(s): s for s in suggestions}.values())

    if not unique_suggestions:
        return {}

    embeddings = model.encode(unique_suggestions)
    kmeans = KMeans(n_clusters=5, random_state=42).fit(embeddings)

    clusters = {}
    for i, label in enumerate(kmeans.labels_):
        clusters.setdefault(f"Cluster {label+1}", []).append(unique_suggestions[i])
    return clusters

# ---------- Streamlit UI ----------

st.set_page_config(page_title="Smart Topic Cluster Generator", layout="wide")
st.title("🧠 AnswerSocrates Clone: Topic Cluster Generator")
st.write("Enter a seed keyword to generate blog topic ideas grouped into intent-based clusters.")

# Session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# Input box
seed = st.text_input("🔍 Enter seed keyword (e.g., home loan, travel insurance, etc.)")

if st.button("Generate Topics"):
    if not seed.strip():
        st.warning("Please enter a keyword.")
    else:
        st.session_state.history.append(seed)
        with st.spinner("Generating clusters..."):
            clusters = generate_clusters(seed)

        if not clusters:
            st.error("No results found. Try a different keyword.")
        else:
            st.success(f"Generated {sum(len(v) for v in clusters.values())} topic ideas across {len(clusters)} clusters.")

            all_topics = []
            output_rows = []
            for cluster_name, topics in clusters.items():
                label = label_cluster(topics)
                st.subheader(f"🔹 Cluster: {label}")
                for topic in topics:
                    intent = get_intent(topic)
                    st.markdown(f"- **{topic}** ({intent})")
                    all_topics.append(topic)
                    output_rows.append({"Cluster": label, "Topic": topic, "Intent": intent})

            # Word cloud
            st.subheader("🔠 Word Cloud of Suggestions")
            show_wordcloud(all_topics)

            # CSV Export
            df = pd.DataFrame(output_rows)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download as CSV",
                csv,
                f"{seed.replace(' ', '_')}_topics.csv",
                "text/csv",
                key='download-csv'
            )

# Sidebar: recent keyword history
st.sidebar.subheader("🔁 Recent Keywords")
for item in st.session_state.history[-5:][::-1]:
    st.sidebar.markdown(f"• {item}")
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Made with ❤️ by <b>Amal Alexander</b></div>",
    unsafe_allow_html=True
)
