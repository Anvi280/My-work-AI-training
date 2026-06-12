from datetime import datetime
from langchain.tools import tool

@tool("get_current_datetime", description="Return the current date and time.")
def get_current_datetime(_: str = "") -> str:
    """Return the current date and time in a readable format."""
    now = datetime.now()
    return now.strftime("Current date and time is %Y-%m-%d %H:%M:%S")
