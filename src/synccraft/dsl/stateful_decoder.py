"""Real XGrammar integration and transactional KV rollback are a future task."""


def decode_program(*args, **kwargs):
    raise NotImplementedError(
        "P1-07: implement tokenizer-aware grammar/symbol masks and atomically "
        "restore token, KV, grammar, AST, symbol, FK and RNG state; keep costs monotonic."
    )
