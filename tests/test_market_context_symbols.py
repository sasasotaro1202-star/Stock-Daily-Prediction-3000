from src.data.market_context import CONTEXT_SYMBOLS


def test_topix_uses_topix_linked_etf_proxy():
    assert CONTEXT_SYMBOLS["topix"] == "1306.T"
