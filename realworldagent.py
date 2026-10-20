import urllib.error
import urllib.request

from langchain.agents import create_agent
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

load_dotenv()

BOOK_URL = "https://www.gutenberg.org/cache/epub/7256/pg7256.txt"

SYSTEM_PROMPT = """You are a literary data assistant.

## Capabilities

- `fetch_text_from_url`: loads the configured Project Gutenberg text. Call it without arguments.
Do not guess line counts or positions—ground them in tool results from the saved file."""


@tool
def fetch_text_from_url() -> str:
    """Fetch the configured Project Gutenberg text."""
    req = urllib.request.Request(
        BOOK_URL,
        headers={"User-Agent": "Mozilla/5.0 (compatible; quickstart-research/1.0)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
    except urllib.error.URLError as e:
        return f"Fetch failed: {e}"
    text = raw.decode("utf-8", errors="replace")
    return text


# Local Qwen via llama.cpp:
# model = init_chat_model(
#     "Qwen",
#     model_provider="openai",
#     base_url="http://127.0.0.1:8080/v1",
#     api_key="local",
#     temperature=0.5,
# )

model = init_chat_model(
    "google_genai:gemini-3.5-flash-lite",
    temperature=0.5,
)

checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[fetch_text_from_url],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)

deep_agent = create_deep_agent(
    model=model,
    tools=[fetch_text_from_url],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)

content = f"""Project Gutenberg hosts a plain-text copy of O. Henry's The Gift of the Magi.
URL: https://www.gutenberg.org/cache/epub/7256/pg7256.txt

Answer as much as you can:

1) How many lines in the complete Gutenberg file contain the substring `Della` (count lines, not occurrences within a line, each line ends with a line break).
2) The 1-based line number of the first line in the file that contains `Jim`.
3) A two-sentence neutral synopsis.

Do your best on (1) and (2). If at any point you realize you cannot **verify** an exact answer with
your available tools and reasoning, do not fabricate numbers: use `null` for that field and spell out
the limitation in `how_you_computed_counts`. If you encounter any errors please report what the error was and what the error message was."""

print("Running create_agent...", flush=True)
agent_result = agent.invoke(
    {"messages": [{"role": "user", "content": content}]},
    config={"configurable": {"thread_id": "great-gatsby-lc"}},
)

print("\ncreate_agent:")
print(agent_result["messages"][-1].content_blocks)


print("Running create_deep_agent...", flush=True)
deep_agent_result = deep_agent.invoke(
    {"messages": [{"role": "user", "content": content}]},
    config={"configurable": {"thread_id": "gift-of-the-magi-da"}},
)


print("\ncreate_deep_agent:")
print(deep_agent_result["messages"][-1].content_blocks)