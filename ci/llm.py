"""One text-completion entry point for every backend.

  adaption/<model-id>   Adaption hosted models via the Responses API (needs ADAPTION_API_KEY)
                        e.g. adaption/qwen3.5-397b, adaption/gemma-4-31b-it, adaption/glm-5.2,
                        adaption/deepseek-v4-flash  (list yours: python -m ci.llm --list)
  anything else         passed to litellm (anthropic/..., openai/..., gemini/...)
"""
from __future__ import annotations

import sys

_adaption_client = None


def _adaption():
    global _adaption_client
    if _adaption_client is None:
        from adaption import Adaption
        _adaption_client = Adaption(timeout=400.0, max_retries=3)  # reads ADAPTION_API_KEY; long generations
    return _adaption_client


def complete(model: str, prompt: str, temperature: float = 0.7, max_tokens: int = 1200) -> tuple[str, dict]:
    """Return (text, usage) where usage has 'in'/'out' token counts when the backend reports them."""
    if model.startswith("adaption/"):
        r = _adaption().responses.create(model=model.split("/", 1)[1], input=prompt,
                                         temperature=temperature, max_output_tokens=max_tokens,
                                         store=False)
        u = getattr(r, "usage", None)
        usage = {"in": getattr(u, "input_tokens", None), "out": getattr(u, "output_tokens", None)} if u else {}
        return r.output_text, usage
    import litellm
    r = litellm.completion(model=model, messages=[{"role": "user", "content": prompt}],
                           temperature=temperature, max_tokens=max_tokens)
    u = getattr(r, "usage", None)
    usage = {"in": u.prompt_tokens, "out": u.completion_tokens} if u else {}
    return r.choices[0].message.content, usage


if __name__ == "__main__":
    if "--list" in sys.argv:
        from ci.run import ROOT  # noqa: F401  (loads .env)
        page = _adaption().models.list(type="hosted")
        for m in getattr(page, "data", page):
            print(getattr(m, "id", m))
    elif len(sys.argv) > 2:  # quick smoke test: python -m ci.llm <model> "<prompt>"
        from ci.run import ROOT  # noqa: F401
        print(complete(sys.argv[1], sys.argv[2], max_tokens=100))
