"""
アメリカ合衆国 50州 + ワシントンD.C. の Markdown を参照するローカル RAG（Streamlit）。
knowledge/*.md のみを読み込みます。質問はプリセットから自動生成（手入力不要）。
"""
from __future__ import annotations

import base64
import os
import random
import subprocess
from pathlib import Path

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="米国各州 RAG", layout="wide")

if not os.environ.get("OPENAI_API_KEY"):
    _env_path = Path(__file__).resolve().parent / ".env"
    if _env_path.exists():
        for line in _env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and not os.environ.get(k):
                    os.environ[k] = v

SCRIPT_DIR = Path(__file__).resolve().parent
STATES_DIR = SCRIPT_DIR / "knowledge"
LOADING_GIF = SCRIPT_DIR / "assets" / "app_demo.gif"
DEMO_USERNAME = os.environ.get("DEMO_USERNAME", "demo_user")
DEMO_PASSWORD = os.environ.get("DEMO_PASSWORD", "demo_pass_123")

RETRIEVAL_K = 6
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

TOPIC_CHOICES: list[tuple[str, str]] = [
    ("summary", "総合（地理・気候・歴史・観光・食・直行便）"),
    ("geo", "地理"),
    ("climate", "気候"),
    ("history", "歴史"),
    ("tourism", "観光"),
    ("food", "食事"),
    ("flights", "日本からの直行便"),
]

SYSTEM_INSTRUCTION = (
    "あなたは人間と話すAIアシスタントです。"
    "以下の参照資料（質問に関連する部分）に基づいて質問に答えてください。"
    "情報について推測は行わず、与えられた情報のみをもとに回答すること。"
    "不明な場合は「文書に記載がありません」と回答すること。"
    "回答の本文は必ず日本語で書いてください。\n\n"
    "参照資料:\n{context}"
)
_chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_INSTRUCTION),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)


def _normalize_line_endings(text: str) -> str:
    if not text:
        return text
    return text.replace("\r\n", "\n").replace("\r", "\n")


def load_state_labels(folder: Path) -> list[tuple[str, str]]:
    """slug と Markdown 先頭見出し（和名）の一覧。"""
    if not folder.is_dir():
        return []
    out: list[tuple[str, str]] = []
    for path in sorted(folder.glob("*.md"), key=lambda p: p.name.lower()):
        text = path.read_text(encoding="utf-8")
        first = text.splitlines()[0] if text else ""
        title = first[2:].strip() if first.startswith("# ") else path.stem
        out.append((path.stem, title))
    return out


def load_markdown_documents(folder: Path) -> list[Document]:
    if not folder.is_dir():
        return []
    docs: list[Document] = []
    for path in sorted(folder.glob("*.md"), key=lambda p: p.name.lower()):
        if not path.is_file():
            continue
        text = _normalize_line_endings(path.read_text(encoding="utf-8"))
        if not text.strip():
            text = "(内容が空でした)"
        docs.append(Document(page_content=text, metadata={"source": path.name}))
    return docs


def build_question(ja_state_name: str, topic_key: str) -> str:
    if topic_key == "summary":
        return (
            f"{ja_state_name}について、地理・気候・歴史・観光・食事・日本からの直行便の各観点を、"
            f"参照資料に基づいて分かりやすくまとめてください。"
        )
    if topic_key == "geo":
        return f"{ja_state_name}の地理について、参照資料に基づいて説明してください。"
    if topic_key == "climate":
        return f"{ja_state_name}の気候について、参照資料に基づいて説明してください。"
    if topic_key == "history":
        return f"{ja_state_name}の歴史について、参照資料に基づいて説明してください。"
    if topic_key == "tourism":
        return f"{ja_state_name}の観光について、参照資料に基づいて説明してください。"
    if topic_key == "food":
        return f"{ja_state_name}の食事・グルメについて、参照資料に基づいて説明してください。"
    if topic_key == "flights":
        return f"{ja_state_name}へ日本人旅行者が行く場合、日本からの直行便の有無・乗り継ぎの傾向について、参照資料に基づいて説明してください。"
    return f"{ja_state_name}について、参照資料に基づいて説明してください。"


@st.cache_resource
def get_rag_components():
    raw_docs = load_markdown_documents(STATES_DIR)
    doc_info = [(d.metadata.get("source", ""), len(d.page_content)) for d in raw_docs]
    if not raw_docs:
        return None, None, 0, 0, []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "、", " ", ""],
    )
    splits = splitter.split_documents(raw_docs)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(splits, embeddings)
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    chain = _chat_prompt | llm
    return vectorstore, chain, len(raw_docs), len(splits), doc_info


if "memory" not in st.session_state:
    st.session_state.memory = []
if "last_question" not in st.session_state:
    st.session_state.last_question = ""
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "last_refs" not in st.session_state:
    st.session_state.last_refs = []
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False


def _build_loading_gif_html(gif_path: Path) -> str:
    if not gif_path.is_file():
        return ""
    gif_b64 = base64.b64encode(gif_path.read_bytes()).decode("ascii")
    return (
        "<div style='display:flex;flex-direction:column;align-items:center;justify-content:center;'>"
        f"<img src='data:image/gif;base64,{gif_b64}' width='180' alt='loading gif' />"
        "<p style='margin-top:8px;'>検索・回答を生成中...</p>"
        "</div>"
    )


def _render_login() -> None:
    st.title("ログイン")
    st.caption("デモアカウントでログインしてください。")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("ユーザーID")
        password = st.text_input("パスワード", type="password")
        submitted = st.form_submit_button("ログイン")
    if submitted:
        if username == DEMO_USERNAME and password == DEMO_PASSWORD:
            st.session_state.authenticated = True
            st.success("ログインに成功しました。")
            st.rerun()
        else:
            st.error("ユーザーIDまたはパスワードが正しくありません。")

if st.session_state.get("file_to_open"):
    try:
        subprocess.run(["open", st.session_state["file_to_open"]], check=False)
    except Exception:
        pass
    del st.session_state["file_to_open"]
if st.session_state.get("folder_to_open"):
    try:
        subprocess.run(["open", st.session_state["folder_to_open"]], check=False)
    except Exception:
        pass
    del st.session_state["folder_to_open"]

state_labels = load_state_labels(STATES_DIR)
vectorstore, chain, num_docs, num_chunks, doc_info = get_rag_components()

st.sidebar.title("米国各州 RAG")
st.sidebar.caption("参照: `knowledge/*.md`（Markdown のみ）")
st.sidebar.write(f"**フォルダ:** `{STATES_DIR}`")
if st.session_state.authenticated:
    st.sidebar.success(f"ログイン中: {DEMO_USERNAME}")
    if st.sidebar.button("ログアウト"):
        st.session_state.authenticated = False
        st.session_state.memory = []
        st.session_state.last_question = ""
        st.session_state.last_answer = ""
        st.session_state.last_refs = []
        st.rerun()
else:
    st.sidebar.info("ログインが必要です。")
    _render_login()
    st.stop()

if st.sidebar.button("🔄 読み込みを更新", help="Markdown を差し替えたあとに押すと再インデックスします"):
    get_rag_components.clear()
    st.rerun()

if num_docs == 0:
    st.sidebar.error("Markdown がありません。`python scripts/gen_us_states_corpus.py` で生成するか、`.md` を置いてください。")
else:
    st.sidebar.success(f"読み込んだ州ファイル: **{num_docs}**")
    st.sidebar.caption(f"チャンク数: {num_chunks}")

st.title("アメリカ各州の特徴を調べる RAG")
st.caption("質問文は下の選択から自動で組み立てます（手入力不要）。回答は参照 Markdown のみに基づきます。")

if not state_labels:
    st.stop()

slug_list = [s for s, _ in state_labels]
ja_list = [j for _, j in state_labels]
idx_default = ja_list.index("カリフォルニア州") if "カリフォルニア州" in ja_list else 0

if "state_pick" not in st.session_state:
    st.session_state.state_pick = idx_default

# ランダムボタン用（ウィジェット key と衝突しない内部状態）
if st.session_state.pop("_do_random_state", False):
    st.session_state.state_pick = random.randint(0, len(ja_list) - 1)

state_index = max(0, min(st.session_state.state_pick, len(ja_list) - 1))

c1, c2, c3 = st.columns([2, 2, 1])
with c1:
    pick = st.selectbox(
        "州（ワシントンD.C.含む）",
        options=list(range(len(ja_list))),
        format_func=lambda i: ja_list[i],
        index=state_index,
    )
    st.session_state.state_pick = pick
with c2:
    topic_labels = [t[1] for t in TOPIC_CHOICES]
    topic_keys = [t[0] for t in TOPIC_CHOICES]
    ti = st.selectbox(
        "トピック",
        options=list(range(len(topic_keys))),
        format_func=lambda i: topic_labels[i],
        index=0,
        key="topic_sb",
    )
with c3:
    st.write("")
    st.write("")
    if st.button("州をランダム", help="デモ用に州だけランダム選択"):
        st.session_state._do_random_state = True
        st.rerun()

slug = slug_list[pick]
ja_name = ja_list[pick]
topic_key = topic_keys[ti]
question = build_question(ja_name, topic_key)

st.info(f"**自動生成された質問:** {question}")

if st.button("この質問で回答を取得", type="primary"):
    if vectorstore is None or chain is None:
        st.error("ベクトルストアを初期化できませんでした。`OPENAI_API_KEY` を確認してください。")
    else:
        loading_placeholder = st.empty()
        if LOADING_GIF.is_file():
            with loading_placeholder.container():
                left, center, right = st.columns([2, 1, 2])
                with center:
                    loading_html = _build_loading_gif_html(LOADING_GIF)
                    if loading_html:
                        st.markdown(loading_html, unsafe_allow_html=True)
        with st.spinner("検索・回答を生成中…"):
            relevant = vectorstore.similarity_search(question, k=RETRIEVAL_K)
            context = "\n\n".join(
                f"[{d.metadata.get('source', '')}]\n{d.page_content}" for d in relevant
            )
            ref_files = list(dict.fromkeys(d.metadata.get("source", "") for d in relevant if d.metadata.get("source")))
            response = chain.invoke({
                "history": st.session_state.memory[-10:],
                "context": context,
                "input": question,
            })
        loading_placeholder.empty()
        st.session_state.last_question = question
        st.session_state.last_answer = response.content
        st.session_state.last_refs = ref_files
        st.session_state.memory.append(HumanMessage(content=question))
        st.session_state.memory.append(AIMessage(content=response.content))
        st.rerun()

if st.session_state.last_answer:
    st.subheader("回答")
    st.write(st.session_state.last_answer)
    refs = st.session_state.last_refs or []
    if refs:
        st.caption("**参照したファイル（検索で取得したチャンク由来）:**")
        for j, fname in enumerate(refs):
            path = (STATES_DIR / fname).resolve()
            col1, col2 = st.columns([4, 1])
            with col1:
                st.text(fname)
                st.caption(str(path))
            with col2:
                if path.is_file():
                    if st.button("開く", key=f"open_ref_{j}"):
                        st.session_state["file_to_open"] = str(path)
                        st.rerun()
    st.divider()
    st.caption("コーパス再生成: `python scripts/gen_us_states_corpus.py`")
