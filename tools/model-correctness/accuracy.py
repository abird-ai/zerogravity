"""Model-independent measurements; source/profile compatibility belongs to callers.

Policies: vLLM db9527a46 tests/models/utils.py:22-47,201-247; llama.cpp
d81235049384534c167caea52b85a694f6103d14 tests/test-backend-ops.cpp:301-315.
Model metrics have no implicit thresholds or cross-profile exactness promise.
"""
import math


def _integer(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite numeric data")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} exceeds finite float range") from error
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _vector(values, label):
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label} must be a nonempty numeric list")
    return [_number(value, label) for value in values]


def _pair(reference, candidate):
    reference = _vector(reference, "reference")
    candidate = _vector(candidate, "candidate")
    if len(reference) != len(candidate):
        raise ValueError("reference/candidate lengths differ")
    difference = [_number(c - r, "difference") for r, c in zip(reference, candidate)]
    return reference, candidate, difference


def _rms(values):
    scale = max(abs(value) for value in values)
    return scale * math.sqrt(math.fsum((value / scale) ** 2 for value in values) / len(values)) if scale else 0.0


def _generation(value, top_k):
    if not isinstance(value, dict) or not isinstance(value.get("text"), str):
        raise ValueError("generation must provide text and token_ids")
    ids = value.get("token_ids")
    if not isinstance(ids, list) or not ids:
        raise ValueError("generation token_ids must be a nonempty list")
    for token in ids:
        _integer(token, "generated token_id")
    if "top_logprobs" not in value:
        raise ValueError("top_logprobs must be explicitly supplied or null")
    rows = value["top_logprobs"]
    if rows is None:
        return ids, value["text"], None
    if not isinstance(rows, list) or len(rows) != len(ids):
        raise ValueError("top_logprobs must cover every generated token")
    returned = []
    for selected, row in zip(ids, rows):
        if not isinstance(row, list) or not top_k <= len(row) <= top_k + 1:
            raise ValueError("top_logprobs row needs top_k entries plus at most one selected extra")
        scores = {}
        for entry in row:
            if not isinstance(entry, dict) or "token_id" not in entry or "logprob" not in entry:
                raise ValueError("top_logprobs entry needs token_id and logprob")
            token = _integer(entry["token_id"], "top_logprobs token_id")
            score = _number(entry["logprob"], "top_logprobs logprob")
            if score > 0:
                raise ValueError("top_logprobs logprob must be nonpositive")
            if token in scores:
                raise ValueError("duplicate top_logprobs token_id")
            scores[token] = score
        if selected not in scores:
            raise ValueError("selected token missing from its own logprobs")
        returned.append(set(scores))
    return ids, value["text"], returned


def compare_generation(reference, candidate, *, policy="vllm-default", top_k=5):
    """Exact IDs+text, or mutual returned-map membership at first divergence.

    The default policy stops semantic comparison at that first divergence;
    all supplied rows are still structurally validated. Equal IDs with unequal
    text are reported but accepted by the upstream default warning policy.
    Returned maps include top-k and any extra selected token; never rerank them.
    """
    if policy not in ("exact", "vllm-default"):
        raise ValueError("unknown generation comparison policy")
    if _integer(top_k, "top_k") == 0:
        raise ValueError("top_k must be positive")
    ref_ids, ref_text, ref_rows = _generation(reference, top_k)
    ids, text, rows = _generation(candidate, top_k)
    if len(ref_ids) != len(ids):
        raise ValueError("generated populations have different token counts")
    first = next((index for index, pair in enumerate(zip(ref_ids, ids)) if pair[0] != pair[1]), None)
    exact_ids, exact_text = first is None, ref_text == text
    mutual = None
    if first is not None and ref_rows is not None and rows is not None:
        mutual = ref_ids[first] in rows[first] and ids[first] in ref_rows[first]
    exact = exact_ids and exact_text
    return {
        "policy": policy, "top_k": top_k, "accepted": exact if policy == "exact" else exact_ids or mutual is True,
        "top_logprobs_semantics": "returned_top_k_plus_selected",
        "exact": exact, "exact_ids": exact_ids, "exact_text": exact_text,
        "mismatch": not exact, "first_divergence": first,
        "matched_prefix_tokens": len(ids) if first is None else first,
        "compared_token_count": len(ids) if first is None else first + 1,
        "numerical_comparison_end": first + 1 if first is not None and mutual is not None else None,
        "mutual_top_k": mutual, "divergent_tail_compared": False,
        "text_warning": policy == "vllm-default" and exact_ids and not exact_text,
    }


def _distribution(logits):
    maximum = max(logits)
    centered = [_number(value - maximum, "centered logit") for value in logits]
    normalizer = math.log(math.fsum(math.exp(value) for value in centered))
    logprobs = [_number(value - normalizer, "logprob") for value in centered]
    return logprobs, [math.exp(value) for value in logprobs]


def full_logits_metrics(reference, candidate, target_id, *, vocabulary_size):
    """Complete aligned vocabulary at ONE identical teacher-forced history.

    delta_nll is also that position's log-perplexity delta; corpus log-PPL
    requires averaging NLL over aligned targets. Top-k data is inadmissible.
    KL retains raw floating roundoff, including possible tiny negative values.
    """
    reference, candidate, difference = _pair(reference, candidate)
    if _integer(vocabulary_size, "vocabulary_size") == 0 or len(reference) != vocabulary_size:
        raise ValueError("logits must contain the declared complete vocabulary")
    if _integer(target_id, "target_id") >= vocabulary_size:
        raise ValueError("target_id exceeds the complete vocabulary")
    ref_logp, ref_p = _distribution(reference)
    logp, probability = _distribution(candidate)
    kl = _number(math.fsum(p * (a - b) for p, a, b in zip(ref_p, ref_logp, logp) if p), "KL")
    ref_nll, nll = -ref_logp[target_id], -logp[target_id]
    return {
        "vocabulary_size": vocabulary_size, "full_vocabulary": True,
        "max_absolute_error": max(map(abs, difference)), "rms_error": _rms(difference),
        "KL_ref_candidate": kl, "reference_nll": ref_nll, "candidate_nll": nll,
        "delta_nll": _number(nll - ref_nll, "delta_nll"),
        "rms_probability_error": _rms([a - b for a, b in zip(ref_p, probability)]),
        "top1_equal": max(range(vocabulary_size), key=reference.__getitem__) == max(range(vocabulary_size), key=candidate.__getitem__),
    }


def backend_nmse(reference, candidate):
    """llama.cpp backend metric: sum((candidate-reference)^2)/sum(candidate^2).

    Denominator is the backend under test; no epsilon or acceptance threshold.
    Zero candidate energy with a nonzero reference is explicitly undefined.
    """
    reference = _vector(reference, "reference")
    candidate = _vector(candidate, "candidate")
    if len(reference) != len(candidate):
        raise ValueError("reference/candidate lengths differ")
    denominator = max(map(abs, candidate))
    if denominator == 0:
        if not any(reference):
            return 0.0
        raise ValueError("NMSE undefined: zero candidate energy and nonzero reference")
    scale = max(denominator, max(map(abs, reference)))
    difference = [(c - r) / scale if math.isfinite(c - r) else c / scale - r / scale
                  for r, c in zip(reference, candidate)]
    numerator_fraction, numerator_exponent = math.frexp(scale)
    denominator_fraction, denominator_exponent = math.frexp(denominator)
    ratio = (numerator_fraction / denominator_fraction) ** 2
    ratio *= math.fsum(value ** 2 for value in difference)
    ratio /= math.fsum((value / denominator) ** 2 for value in candidate)
    try:
        return _number(math.ldexp(ratio, 2 * (numerator_exponent - denominator_exponent)), "NMSE")
    except OverflowError as error:
        raise ValueError("NMSE exceeds finite float range") from error
