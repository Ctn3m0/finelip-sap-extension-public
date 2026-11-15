"""Text preprocessing for the FineLIP + SAP experiment."""

import random


def _with_period(text):
    text = text.strip()
    if text and not text.endswith("."):
        text += "."
    return text


def construct_subcaptions(
    long_caption,
    short_caption,
    max_subcaptions=15,
    merge_probability=0.5,
    rng=None,
):
    """Construct SAP subcaptions using the supplied MulCLIP training policy.

    The first sentence is excluded because the reference code treats it as the
    short summary. Remaining adjacent sentences are optionally merged, short
    fragments are removed, and the short caption is used as a safe fallback.
    """
    if max_subcaptions < 1:
        raise ValueError("max_subcaptions must be at least 1")
    if not 0.0 <= merge_probability <= 1.0:
        raise ValueError("merge_probability must be between 0 and 1")

    rng = rng or random
    sentences = [part.strip() for part in (long_caption or "").split(".") if part.strip()]
    results = []
    index = 1
    while index < len(sentences):
        current = sentences[index]
        if index + 1 < len(sentences) and rng.random() < merge_probability:
            results.append(_with_period(f"{current}. {sentences[index + 1]}"))
            index += 2
        else:
            results.append(_with_period(current))
            index += 1

    results = [
        item for item in results
        if len(item) >= 15 and not (len(item) == 2 and item[0].isalpha())
    ]
    if len(results) > max_subcaptions:
        results = rng.sample(results, max_subcaptions)
    if not results:
        fallback = _with_period(short_caption or "")
        if not fallback:
            fallback = _with_period(long_caption or "")
        if not fallback:
            raise ValueError("SAP requires a non-empty long or short caption")
        results = [fallback]
    return results
