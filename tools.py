"""Tool definitions available to the LangGraph agent."""
import ast
import math
import os
import operator
import requests
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.types import interrupt

from rag import get_retriever


search_tool = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced",
)


@tool
def rag_tool(query: str) -> str:
    """Retrieve relevant information from the PDF uploaded to this app."""
    try:
        documents = get_retriever().invoke(query)
    except FileNotFoundError as error:
        return str(error)

    if not documents:
        return "No relevant information was found in the PDF."

    results = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page", "Unknown page")
        results.append(
            f"Document {index}\nSource: {source}\nPage: {page}\n"
            f"Content: {document.page_content}"
        )
    return "\n\n".join(results)


# A small arithmetic parser avoids evaluating arbitrary Python expressions.
_ALLOWED_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_ALLOWED_FUNCTIONS = {
    "sqrt": math.sqrt, "ceil": math.ceil, "floor": math.floor,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "log": math.log,
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
}


def _safe_math(node):
    if isinstance(node, ast.Expression):
        return _safe_math(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINARY:
        left, right = _safe_math(node.left), _safe_math(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("Exponent is too large.")
        return _ALLOWED_BINARY[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_safe_math(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        function = _ALLOWED_FUNCTIONS.get(node.func.id)
        if function is None or node.keywords:
            raise ValueError("That function is not allowed.")
        return function(*[_safe_math(arg) for arg in node.args])
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_safe_math(item) for item in node.elts]
    raise ValueError("Expression contains an unsupported operation.")


@tool
def calculator(expression: str) -> str:
    """Calculate arithmetic expressions; supports sqrt(16), 2 + 2, and similar."""
    try:
        parsed = ast.parse(expression, mode="eval")
        return str(_safe_math(parsed))
    except Exception as error:
        return f"Calculation error: {error}"


@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch a stock quote from Alpha Vantage for a ticker such as AAPL or TSLA."""
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        return {"error": "Missing ALPHA_VANTAGE_API_KEY in the .env file."}
    try:
        response = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "GLOBAL_QUOTE",
                "symbol": symbol.strip().upper(),
                "apikey": api_key,
            },
            timeout=15,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as error:
        return {"error": f"Stock quote request failed: {error}"}


@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """Simulate a stock purchase, pausing for human approval before confirming."""
    if quantity <= 0:
        return {"status": "cancelled", "message": "Quantity must be greater than zero."}

    decision = interrupt(
        f"Approve simulated purchase of {quantity} shares of {symbol.upper()}? (yes/no)"
    )
    if isinstance(decision, str) and decision.strip().lower() == "yes":
        return {
            "status": "success",
            "message": f"Simulated purchase order placed for {quantity} shares of {symbol.upper()}.",
            "symbol": symbol.upper(),
            "quantity": quantity,
        }
    return {
        "status": "cancelled",
        "message": f"Purchase of {quantity} shares of {symbol.upper()} was declined.",
        "symbol": symbol.upper(),
        "quantity": quantity,
    }


@tool
def get_current_weather(location: str) -> str:
    """Get current weather for a city or location using OpenWeatherMap."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Weather API key is missing. Set OPENWEATHER_API_KEY in .env."

    try:
        geo_response = requests.get(
            "https://api.openweathermap.org/geo/1.0/direct",
            params={"q": location, "limit": 1, "appid": api_key},
            timeout=10,
        )
        geo_response.raise_for_status()
        locations = geo_response.json()
        if not locations:
            return f"Could not find the location: {location}"

        place = locations[0]
        weather_response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={
                "lat": place["lat"],
                "lon": place["lon"],
                "appid": api_key,
                "units": "metric",
            },
            timeout=10,
        )
        weather_response.raise_for_status()
        data = weather_response.json()

        location_parts = [
            place.get("name", location),
            place.get("state"),
            place.get("country"),
        ]
        display_location = ", ".join(part for part in location_parts if part)
        visibility = data.get("visibility")
        visibility_km = round(visibility / 1000, 1) if visibility is not None else "N/A"

        return (
            f"Current weather in {display_location}:\n"
            f"- Condition: {data['weather'][0]['description'].title()}\n"
            f"- Temperature: {data['main']['temp']}°C\n"
            f"- Feels like: {data['main']['feels_like']}°C\n"
            f"- Humidity: {data['main']['humidity']}%\n"
            f"- Pressure: {data['main']['pressure']} hPa\n"
            f"- Wind speed: {data.get('wind', {}).get('speed', 'N/A')} m/s\n"
            f"- Visibility: {visibility_km} km"
        )
    except requests.Timeout:
        return "The weather service request timed out. Please try again."
    except requests.HTTPError as error:
        status = error.response.status_code if error.response is not None else "unknown"
        if status == 401:
            return "The OpenWeatherMap API key is invalid or inactive."
        return f"Weather API returned an HTTP error: {status}"
    except requests.RequestException as error:
        return f"Could not connect to the weather service: {error}"
    except (KeyError, TypeError, ValueError) as error:
        return f"Unexpected weather API response: {error}"


TOOLS = [
    search_tool,
    calculator,
    get_stock_price,
    get_current_weather,
    rag_tool,
    purchase_stock,
]
