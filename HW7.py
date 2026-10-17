"""
build_db.py  --  the whole news bot as ONE page of a multipage Streamlit app.

Put this file in your pages/ folder (or register it with st.Page) and keep
news.csv in the folder you run `streamlit run app.py` from.

First visit: the ChromaDB database does not exist yet, so this page builds it
             automatically (cleans news.csv, scores each article's
             interestingness with an LLM, embeds everything). A few minutes,
             and it runs ONCE.
Later visits: the database already exists, so the page just loads it.

API key: OPEN_AI_KEY in .streamlit/secrets.toml (or the OPENAI_API_KEY env var).
"""
import sys

try:  # Streamlit Cloud / some Codespaces need pysqlite3 for ChromaDB
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import difflib
import hashlib
import json
import os
import re
from datetime import date, datetime
from urllib.parse import urlparse

import numpy as np
import pandas as pd

CSV_PATH = "news.csv"
DB_PATH = "./ChromaDB_for_News"
COLLECTION_NAME = "NewsCollection"
SCORES_CACHE = "scores_cache.json"

EMBED_MODEL = "text-embedding-3-small"
SCORE_MODEL = "gpt-4o-mini"
CHAT_MODEL = "gpt-4o-mini"

# Coverage bonus tuning: two articles count as "the same story" above this
# cosine similarity. NOT validated against your data -- run with
# --show-clusters, eyeball the top stories, and adjust.
SAME_STORY_SIM = 0.80
COVERAGE_BONUS_PER_ARTICLE = 0.5
COVERAGE_BONUS_MAX = 1.5


# ==========================================================================
# PART 1: BUILD THE DATABASE
# ==========================================================================
def clean_company(name: str) -> str:
    # Some names carry trailing spaces or non-breaking spaces ("Cisco\xa0")
    return re.sub(r"\s+", " ", str(name).replace("\xa0", " ")).strip()


def split_title(document: str) -> str:
    # Documents look like: "<headline> Description: <snippet>"
    return document.split(" Description:", 1)[0].strip()


def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["company_name"] = df["company_name"].map(clean_company)
    df["Document"] = df["Document"].astype(str).str.strip()
    df["URL"] = df["URL"].astype(str).str.strip()

    # Dates come in two formats ("...+00:00" and "...Z"); ISO8601 handles both
    df["Date"] = pd.to_datetime(df["Date"], format="ISO8601", utc=True)

    before = len(df)
    df = df.drop_duplicates(subset=["company_name", "URL"]).reset_index(drop=True)
    print(f"Loaded {before} rows -> {len(df)} after dropping exact duplicates")

    df["title"] = df["Document"].map(split_title)
    df["source"] = df["URL"].map(
        lambda u: re.sub(r"^www\.", "", urlparse(u).netloc) or "unknown"
    )
    # Stable ID so re-running the build doesn't change IDs
    df["id"] = df.apply(
        lambda r: hashlib.sha1(f"{r.company_name}|{r.URL}".encode()).hexdigest()[:12],
        axis=1,
    )
    return df


SCORING_PROMPT = """You are a financial and technology news editor. For each article below,
judge how INTERESTING it is to someone following the named company.

Score 1-10:
  9-10  Major, market-moving or strategy-changing news: large M&A or funding,
        earnings surprises, regulatory/legal action, leadership changes,
        major product launches, big layoffs, security breaches.
  6-8   Clearly notable: meaningful partnerships, notable product news, analyst
        upgrades/downgrades with substance, significant contracts.
  3-5   Routine or minor: incremental updates, commentary, minor announcements.
  1-2   Boilerplate (e.g. "total voting rights" notices), listicles, generic
        stock-price chatter, or very little substance.

Also set "on_topic" to false if the article is NOT really about the named company
as a business (e.g. same word, different thing: a neighbourhood, a person, a
podcast that merely mentions the company). Off-topic articles get score 1.

"why" must be ONE sentence explaining why a reader should care, using only facts
in the article text. Do not invent details.

Respond with JSON only:
{"results": [{"id": "...", "score": 7, "on_topic": true, "why": "..."}]}
"""


def score_articles(client, df: pd.DataFrame, batch_size: int = 15) -> dict:
    cache = {}
    if os.path.exists(SCORES_CACHE):
        with open(SCORES_CACHE) as f:
            cache = json.load(f)

    todo = df[~df["id"].isin(cache)]
    print(f"Scoring {len(todo)} articles ({len(cache)} already cached)...")

    for start in range(0, len(todo), batch_size):
        batch = todo.iloc[start : start + batch_size]
        batch_ids = set(batch["id"])
        payload = [
            {
                "id": r.id,
                "company": r.company_name,
                "date": r.Date.strftime("%Y-%m-%d"),
                "text": r.Document[:700],
            }
            for r in batch.itertuples()
        ]
        try:
            resp = client.chat.completions.create(
                model=SCORE_MODEL,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SCORING_PROMPT},
                    {"role": "user", "content": json.dumps(payload)},
                ],
            )
            results = json.loads(resp.choices[0].message.content)["results"]
            for item in results:
                if item.get("id") in batch_ids:
                    cache[item["id"]] = {
                        "score": float(max(1, min(10, item.get("score", 3)))),
                        "on_topic": bool(item.get("on_topic", True)),
                        "why": str(item.get("why", "")).strip(),
                    }
        except Exception as e:  # keep going; failed batches are retried next run
            print(f"  batch at {start} failed: {e}", file=sys.stderr)

        with open(SCORES_CACHE, "w") as f:
            json.dump(cache, f)
        print(f"  scored {min(start + batch_size, len(todo))}/{len(todo)}")

    return cache


def embed_texts(client, texts, batch_size=100):
    vectors = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=chunk)
        vectors.extend(d.embedding for d in resp.data)
        print(f"  embedded {min(start + batch_size, len(texts))}/{len(texts)}")
    return np.array(vectors, dtype=np.float32)


def coverage_counts(df: pd.DataFrame, vectors: np.ndarray) -> np.ndarray:
    """How many *other* articles (different URL) cover the same story?"""
    norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    sims = norm @ norm.T
    np.fill_diagonal(sims, 0)
    urls = df["URL"].to_numpy()
    different_url = urls[:, None] != urls[None, :]
    return ((sims >= SAME_STORY_SIM) & different_url).sum(axis=1)


def database_exists() -> bool:
    import chromadb
    if not os.path.isdir(DB_PATH):
        return False
    try:
        col = chromadb.PersistentClient(path=DB_PATH).get_collection(COLLECTION_NAME)
        return col.count() > 0
    except Exception:
        return False


def build_database(client, csv_path=CSV_PATH, show_clusters=False):
    import chromadb

    df = load_and_clean(csv_path)
    scores = score_articles(client, df)

    # Anything that failed to score gets a neutral default
    df["llm_score"] = df["id"].map(lambda i: scores.get(i, {}).get("score", 3.0))
    df["on_topic"] = df["id"].map(lambda i: scores.get(i, {}).get("on_topic", True))
    df["why"] = df["id"].map(lambda i: scores.get(i, {}).get("why", ""))

    print("Embedding articles...")
    vectors = embed_texts(
        client, [f"{r.company_name}: {r.Document[:2000]}" for r in df.itertuples()]
    )

    df["coverage"] = coverage_counts(df, vectors)
    bonus = np.minimum(COVERAGE_BONUS_MAX, df["coverage"] * COVERAGE_BONUS_PER_ARTICLE)
    df["interest_score"] = np.round(np.minimum(10, df["llm_score"] + bonus), 2)

    if show_clusters:
        print(df.sort_values("coverage", ascending=False)
                .head(15)[["company_name", "title", "coverage"]].to_string())

    chroma = chromadb.PersistentClient(path=DB_PATH)
    try:
        chroma.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = chroma.create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    metadatas = [
        {
            "company": r.company_name,
            "title": r.title,
            "url": r.URL,
            "source": r.source,
            "date": r.Date.strftime("%Y-%m-%d"),
            # given in the CSV; lets us filter on date ranges numerically
            "days_since_2000": int(r.days_since_2000),
            "interest_score": float(r.interest_score),
            "coverage": int(r.coverage),
            "on_topic": bool(r.on_topic),
            "why": r.why,
        }
        for r in df.itertuples()
    ]

    for start in range(0, len(df), 500):
        end = start + 500
        collection.add(
            ids=df["id"].tolist()[start:end],
            documents=df["Document"].tolist()[start:end],
            embeddings=vectors[start:end].tolist(),
            metadatas=metadatas[start:end],
        )

    print(f"Done: {collection.count()} articles in {DB_PATH}/{COLLECTION_NAME}")
    print(f"Off-topic flagged: {(~df['on_topic']).sum()}")
    print("Top 10 by interest score:")
    print(df.sort_values("interest_score", ascending=False)
            .head(10)[["company_name", "title", "interest_score"]].to_string())


# ==========================================================================
# PART 2: THE CHATBOT (Streamlit)
# ==========================================================================
# Names people use that differ from how the dataset labels the company
ALIASES = {
    "meta": "Facebook", "google": "Alphabet",
    "tsmc": "Taiwan Semiconductor Manufacturing Co., Ltd. (TSMC)",
    "amd": "Advanced Micro Devices (AMD)", "disney": "Walt Disney", "sony": "SONY",
    "samsung": "Samsung Electronics", "jpmorgan": "JPMorgan Chase",
    "jp morgan": "JPMorgan Chase", "hp inc": "HP",
    "hpe": "Hewlett Packard Enterprise", "ebay": "eBay Inc.",
}

# Filled in by the page code below once the database is loaded
COMPANIES = []


def resolve_company(name):
    """Map what the user typed to the dataset's company label (or None)."""
    if not name:
        return None
    key = name.strip().lower()
    if key in ALIASES:
        return ALIASES[key]
    lowered = {c.lower(): c for c in COMPANIES}
    if key in lowered:
        return lowered[key]
    contains = [c for lc, c in lowered.items() if key in lc or lc in key]
    if len(contains) == 1:
        return contains[0]
    close = difflib.get_close_matches(key, list(lowered), n=1, cutoff=0.7)
    return lowered[close[0]] if close else None


def to_days_since_2000(date_str):
    d = datetime.strptime(date_str, "%Y-%m-%d").date()
    return (d - date(2000, 1, 1)).days


def build_where(company=None, start_date=None, end_date=None):
    # Off-topic articles (e.g. a neighbourhood that shares a company's name) are excluded.
    conditions = [{"on_topic": True}]
    if company:
        conditions.append({"company": company})
    if start_date:
        conditions.append({"days_since_2000": {"$gte": to_days_since_2000(start_date)}})
    if end_date:
        conditions.append({"days_since_2000": {"$lte": to_days_since_2000(end_date)}})
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def format_article(meta, doc, extra=None):
    snippet = doc if len(doc) <= 500 else doc[:500] + "..."
    item = {
        "company": meta["company"],
        "title": meta["title"],
        "date": meta["date"],
        "source": meta["source"],
        "url": meta["url"],
        "snippet": snippet,
    }
    if extra:
        item.update(extra)
    return item


def company_error(company):
    suggestions = difflib.get_close_matches(
        company.lower(), [c.lower() for c in COMPANIES], n=5, cutoff=0.4)
    return (f"No company matching '{company}' in this dataset. "
            f"Closest names: {suggestions if suggestions else 'none'}. "
            f"The dataset covers {len(COMPANIES)} companies.")


# --- Tool 1: semantic search for a company/topic ---
def search_news(query, collection, client, company=None, start_date=None,
                end_date=None, n_results=5):
    resolved = None
    if company:
        resolved = resolve_company(company)
        if resolved is None:
            return company_error(company)

    response = client.embeddings.create(
        input=f"{resolved}: {query}" if resolved else query,
        model=EMBED_MODEL,
    )
    query_embedding = response.data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=build_where(resolved, start_date, end_date),
        include=["documents", "metadatas", "distances"],
    )

    docs = results['documents'][0]
    if not docs:
        return "No relevant articles were found."

    items = [
        format_article(meta, doc, {
            "similarity": round(1 - dist, 3),
            "interest_score": meta["interest_score"],
            "why_it_matters": meta["why"],
        })
        for doc, meta, dist in zip(docs, results['metadatas'][0], results['distances'][0])
    ]
    return json.dumps(items, indent=1)


# --- Tool 2: ranked list of the most interesting articles ---
def most_interesting_news(collection, company=None, start_date=None,
                          end_date=None, n=5):
    resolved = None
    if company:
        resolved = resolve_company(company)
        if resolved is None:
            return company_error(company)

    results = collection.get(
        where=build_where(resolved, start_date, end_date),
        include=["documents", "metadatas"],
    )
    if not results['ids']:
        return "No articles matched those filters."

    rows = list(zip(results['documents'], results['metadatas']))
    # Highest interest first; ties broken by wider coverage, then recency
    rows.sort(key=lambda r: (r[1]["interest_score"], r[1]["coverage"], r[1]["date"]),
              reverse=True)

    items = []
    for rank, (doc, meta) in enumerate(rows[:n], start=1):
        items.append(format_article(meta, doc, {
            "rank": rank,
            "interest_score": meta["interest_score"],
            "why_it_matters": meta["why"],
            "other_articles_on_same_story": meta["coverage"],
        }))
    return json.dumps(items, indent=1)


# JSON schemas describing the tools to the OpenAI API
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": (
                "Search the news articles for a specific company or topic and return the most "
                "relevant ones. Use for requests like 'find news about X', where X is a company "
                "(pass it as `company`) or a topic such as 'AI chips' (pass it only in `query`)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for, e.g. 'earnings' or 'AI chip demand'."
                    },
                    "company": {
                        "type": "string",
                        "description": "Optional. Restrict results to this company, e.g. 'Nvidia'."
                    },
                    "start_date": {"type": "string", "description": "Optional. YYYY-MM-DD, inclusive."},
                    "end_date": {"type": "string", "description": "Optional. YYYY-MM-DD, inclusive."},
                    "n_results": {"type": "integer", "description": "How many articles to return (default 5, max 10)."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "most_interesting_news",
            "description": (
                "Return a ranked list (most interesting first) of articles, each with a rationale for "
                "why it matters. Use for 'find the most interesting news', 'top stories', or "
                "'biggest news for <company>'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {"type": "string", "description": "Optional. Only rank this company's articles."},
                    "start_date": {"type": "string", "description": "Optional. YYYY-MM-DD, inclusive."},
                    "end_date": {"type": "string", "description": "Optional. YYYY-MM-DD, inclusive."},
                    "n": {"type": "integer", "description": "How many to return (default 5, max 15)."}
                },
                "required": []
            }
        }
    }
]


def run_tool(name, args, collection, client):
    try:
        if name == "search_news":
            return search_news(
                args.get("query", ""), collection, client,
                company=args.get("company"),
                start_date=args.get("start_date"),
                end_date=args.get("end_date"),
                n_results=max(1, min(int(args.get("n_results", 5)), 10)),
            )
        if name == "most_interesting_news":
            return most_interesting_news(
                collection,
                company=args.get("company"),
                start_date=args.get("start_date"),
                end_date=args.get("end_date"),
                n=max(1, min(int(args.get("n", 5)), 15)),
            )
    except ValueError as e:  # e.g. a malformed date from the model
        return f"Invalid argument: {e}. Dates must be YYYY-MM-DD."
    return ""


def get_buffered_messages(messages, max_messages=5):
    system_msgs = [m for m in messages if m["role"] == "system"]
    user_idxs = [i for i, m in enumerate(messages) if m["role"] == "user"][-max_messages:]
    assistant_idxs = [i for i, m in enumerate(messages) if m["role"] == "assistant"][-max_messages:]

    keep_idxs = sorted(set(user_idxs + assistant_idxs))
    trimmed = [messages[i] for i in keep_idxs]

    return system_msgs + trimmed


# ==========================================================================
# THE PAGE (runs top to bottom every time Streamlit reruns)
# ==========================================================================
import streamlit as st
from openai import OpenAI
import chromadb

st.title("News Reporting Bot")

if 'openai_client' not in st.session_state:
    api_key = st.secrets["OPEN_AI_KEY"]
    st.session_state.openai_client = OpenAI(api_key=api_key)
client = st.session_state.openai_client

# --- Build the database ONCE, the first time this page is opened ---
if not database_exists():
    with st.spinner("First run: building the news database (a few minutes, happens once)..."):
        build_database(client)

chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_collection(COLLECTION_NAME)

# --- Dataset facts, read from the DB so the prompt and tools stay accurate ---
all_meta = collection.get(include=["metadatas"])["metadatas"]
COMPANIES = sorted({m["company"] for m in all_meta})
dates = sorted(m["date"] for m in all_meta)
min_date, max_date, n_articles = dates[0], dates[-1], len(all_meta)

SYSTEM_PROMPT = f"""You are a news reporting assistant. You ONLY report on a fixed set of {n_articles} articles
covering {len(COMPANIES)} companies, published between {min_date} and {max_date}.
So "latest", "today" or "this week" all mean within that window.

You have two tools:
- search_news: find articles about a specific company or topic.
- most_interesting_news: return a ranked list of the most interesting articles, optionally for one company or date range.

Rules:
- Always call a tool before answering a question about the news. Never answer news questions from memory.
- Use ONLY what the tool results say. Never add facts, numbers, or quotes that are not in the results.
- For every article you mention, give the company, the date, and the source link.
- For "most interesting" requests, present a numbered list in the order the tool returned it, and for each item give
  a short "Why it matters" using the tool's rationale. You may add one sentence of context if related articles in the
  results support it (for example, the same story covered by several outlets).
- If the tool returns nothing relevant, say so plainly and suggest a nearby company or topic. Do not fill the gap
  with outside knowledge.
- If the user's question is not about the news in this dataset, briefly say what you can help with.
- Keep answers concise and easy to scan.
"""

# NOTE: session_state is shared by every page, so this page uses its own key
if "news_messages" not in st.session_state:
    st.session_state["news_messages"] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": "What news can I help you find? Try \"Find the most interesting news\" or \"Find news about Nvidia\"."}
    ]

for msg in st.session_state.news_messages:
    if msg["role"] == "system":
        continue
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])

if prompt := st.chat_input("Ask about the news..."):
    st.session_state.news_messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    buffered_messages = get_buffered_messages(st.session_state.news_messages)

    # --- Step 1: let the LLM decide which tool(s) to call ---
    first_response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=buffered_messages,
        tools=TOOLS,
        tool_choice="auto"
    )

    first_msg = first_response.choices[0].message
    tool_calls = first_msg.tool_calls

    if tool_calls:
        messages_with_tool_result = buffered_messages + [
            {
                "role": "assistant",
                "content": first_msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in tool_calls
                ]
            }
        ]

        for tc in tool_calls:
            args = json.loads(tc.function.arguments)
            tool_output = run_tool(tc.function.name, args, collection, client)

            messages_with_tool_result.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_output
            })

        # --- Step 2: give the LLM the tool results and let it answer.
        stream = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages_with_tool_result,
            stream=True
        )
    else:
        # The model chose to answer without retrieving anything.
        stream = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=buffered_messages,
            stream=True
        )

    with st.chat_message("assistant"):
        response = st.write_stream(stream)

    st.session_state.news_messages.append({"role": "assistant", "content": response})