from src.data.market_context import CONTEXT_SYMBOLS


def test_topix_uses_current_yahoo_symbol():
    assert CONTEXT_SYMBOLS["topix"] == "^TOPX"
