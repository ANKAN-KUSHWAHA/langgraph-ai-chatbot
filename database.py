"""SQLite checkpoint persistence for LangGraph."""
import sqlite3
from config import SQLITE_PATH
from langgraph.checkpoint.sqlite import SqliteSaver

# The absolute project-local path prevents a different working directory from
# accidentally selecting another project's chatbot.sqlite file.
_connection = sqlite3.connect(
    str(SQLITE_PATH),
    check_same_thread=False,
)
checkpointer = SqliteSaver(_connection)


def get_all_threads() -> list[str]:
    """Return unique thread IDs saved in this project's checkpoint database."""
    return sorted({
        checkpoint.config["configurable"]["thread_id"]
        for checkpoint in checkpointer.list(None)
        if checkpoint.config
        and checkpoint.config.get("configurable", {}).get("thread_id")
    })
