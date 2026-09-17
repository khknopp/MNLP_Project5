"""
evaluation.py - Evaluate saved translations with BLEU, chrF or COMET.

Import evaluate_direction or evaluate_all_directions into a notebook to score
translations and return results for tables, comparisons and plots.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DIRECTIONS = ("az-en", "en-az", "be-en", "en-be")
COMET_MODEL = "Unbabel/wmt22-comet-da"


def _validate_metric(metric_name):
    """
    Normalize a metric name and check that it is supported.

    Args:
        metric_name (str): 'bleu', 'chrf' or 'comet'. Capitalization and
            surrounding whitespace are ignored.

    Returns:
        str: The validated, lowercase metric name.

    Raises:
        ValueError: If the metric name is unsupported.
    """
    name = metric_name.strip().lower()

    if name not in ("bleu", "chrf", "comet"):
        raise ValueError(f"Unknown metric: {metric_name!r}. Choose bleu, chrf or comet.")

    return name


def _translation_paths(folder, direction, include_source):
    """
    Build the required file paths and check that each file exists.

    Args:
        folder (str or Path): Translation folder. Relative paths start at
            the repository root, regardless of the notebook's directory.
        direction (str): Translation direction, such as 'en-be'.
        include_source (bool): Whether to require the .src file as well
            as the .ref and .hyp files.

    Returns:
        dict[str, Path]: Paths under 'ref' and 'hyp', plus 'src' when
            include_source is True. For example, 'hyp' points to en-be.hyp.

    Raises:
        FileNotFoundError: If a required file does not exist.
    """
    folder = Path(folder)

    if not folder.is_absolute():
        folder = ROOT / folder

    if include_source:
        extensions = ("src", "ref", "hyp")
    else:
        extensions = ("ref", "hyp")

    paths = {}

    for extension in extensions:
        path = folder / f"{direction}.{extension}"

        if not path.is_file():
            raise FileNotFoundError(f"Missing translation file: {path}")

        paths[extension] = path

    return paths


def load_translations(folder, direction, include_source=True):
    """
    Read aligned translation files and check their line counts.

    Each line is one example. Empty lines and whitespace are preserved;
    only line endings are removed, so sentence positions stay aligned.

    Args:
        folder (str or Path): Translation folder, absolute or relative to
            the repository root.
        direction (str): Translation direction, such as 'en-be'.
        include_source (bool): Read .src in addition to .ref and .hyp.
            Defaults to True; BLEU and chrF do not need source sentences.

    Returns:
        dict[str, list[str]]: 'hyp' contains model translations and 'ref'
            contains human references. 'src' contains original sentences
            when include_source is True. Lists retain the file order.

    Raises:
        FileNotFoundError: If a required file does not exist.
        ValueError: If files have unequal line counts or contain no lines.
    """
    paths = _translation_paths(folder, direction, include_source)
    translations = {}

    for extension, path in paths.items():
        with path.open(encoding="utf-8") as file:
            translations[extension] = [line.rstrip("\r\n") for line in file]

    # Every line must have a matching line in the other files.
    counts = {key: len(lines) for key, lines in translations.items()}

    if len(set(counts.values())) != 1:
        raise ValueError(f"{direction}: unequal line counts: {counts}")

    if counts["hyp"] == 0:
        raise ValueError(f"{direction}: translation files are empty")

    return translations


def load_metric(metric_name, *, beta=2, word_order=0):
    """
    Create a metric that can be reused across evaluation calls.

    COMET downloads its checkpoint on first use. BLEU and chrF use
    SacreBLEU and do not load a neural model.

    Args:
        metric_name (str): 'bleu', 'chrf' or 'comet', case-insensitive.
        beta (float): chrF recall weighting relative to precision.
            Defaults to 2 (chrF2); 1 balances precision and recall.
            Keyword-only; ignored for BLEU and COMET.
        word_order (int): Maximum word n-gram length added to chrF.
            Defaults to 0 (characters only); 2 enables chrF++.
            Keyword-only; ignored for BLEU and COMET.

    Returns:
        BLEU, CHRF or COMET model: A ready-to-use evaluator for the
            requested metric. Pass it as metric to the evaluation functions.

    Raises:
        ValueError: If the metric name is unsupported.
        ImportError: If sacrebleu or unbabel-comet is not installed for
            the selected metric.
    """
    metric_name = _validate_metric(metric_name)

    if metric_name == "comet":
        from comet import download_model, load_from_checkpoint

        model_path = download_model(COMET_MODEL)
        return load_from_checkpoint(model_path)

    from sacrebleu.metrics import BLEU, CHRF

    if metric_name == "bleu":
        return BLEU()

    # default beta=2 selects chrF2; word_order=0 excludes word-level n-grams.
    return CHRF(beta=beta, word_order=word_order)


def _score_translations(
    translations,
    direction,
    metric_name,
    metric,
    batch_size,
    gpus,
):
    """Calculate scores using already validated texts and a loaded metric.

    Args:
        translations (dict[str, list[str]]): Output of load_translations.
            COMET requires 'src', 'hyp' and 'ref'; BLEU/chrF need the latter two.
        direction (str): Direction label to include in the result.
        metric_name (str): Validated lowercase metric name.
        metric (object): Loaded evaluator matching metric_name.
        batch_size (int): Sentences per COMET inference batch. Ignored
            for BLEU and chrF.
        gpus (int): GPUs to use for COMET; 0 selects CPU. Ignored for
            BLEU and chrF.

    Returns:
        dict: 'direction', 'metric', 'n_sentences' and 'score' describe
            the corpus result. COMET adds 'sentence_scores' in input order.
            BLEU/chrF add 'signature', recording their settings and version.
    """
    hypotheses = translations["hyp"]
    references = translations["ref"]

    result = {
        "direction": direction,
        "metric": metric_name,
        "n_sentences": len(hypotheses),
    }

    if metric_name == "comet":
        sources = translations["src"]
        examples = []

        # COMET expects one dictionary per sentence. 'mt' is its name for
        # the machine translation (our hypothesis).
        for source, hypothesis, reference in zip(sources, hypotheses, references):
            examples.append({
                "src": source,
                "mt": hypothesis,
                "ref": reference,
            })

        prediction = metric.predict(
            examples,
            batch_size=batch_size,
            gpus=gpus,
        )

        result["score"] = float(prediction.system_score)
        result["sentence_scores"] = [float(score) for score in prediction.scores]

    else:
        # Corpus BLEU/chrF must not be calculated by averaging sentence scores.
        corpus_result = metric.corpus_score(hypotheses, [references])

        result["score"] = float(corpus_result.score)
        result["signature"] = str(metric.get_signature())

    return result


def evaluate_direction(
    folder,
    direction,
    metric_name,
    metric=None,
    batch_size=8,
    gpus=0,
):
    """Read and evaluate one translation direction.

    Validate the metric and input texts before loading an evaluator.
    Compare scores only within the same metric and configuration.

    Args:
        folder (str or Path): Translation folder, absolute or relative to
            the repository root.
        direction (str): Translation direction, such as 'en-be'.
        metric_name (str): 'bleu', 'chrf' or 'comet', case-insensitive.
        metric (object, optional): Evaluator returned by load_metric;
            must match metric_name. If None, load one with default settings.
        batch_size (int): Sentences per COMET batch. Defaults to 8;
            ignored for BLEU and chrF.
        gpus (int): GPUs for COMET. Defaults to 0 (CPU); use 1 for one
            GPU. Ignored for BLEU and chrF.

    Returns:
        dict: Contains 'direction', 'metric', 'n_sentences' and 'score'.
            COMET also returns 'sentence_scores' in file order. BLEU/chrF
            return a 'signature' describing the metric settings and version.

    Raises:
        ValueError: If the metric is unsupported, line counts differ,
            or the required files contain no lines.
        FileNotFoundError: If a required translation file is missing.
        ImportError: If a required metric package is unavailable when loading.
    """
    metric_name = _validate_metric(metric_name)

    translations = load_translations(
        folder,
        direction,
        include_source=(metric_name == "comet"),
    )

    # A notebook can supply a loaded metric to reuse it across calls.
    if metric is None:
        metric = load_metric(metric_name)

    return _score_translations(
        translations=translations,
        direction=direction,
        metric_name=metric_name,
        metric=metric,
        batch_size=batch_size,
        gpus=gpus,
    )


def evaluate_all_directions(
    folder,
    metric_name,
    metric=None,
    directions=DIRECTIONS,
    batch_size=8,
    gpus=0,
):
    """Evaluate multiple directions from one system's translation folder.

    Check all required paths up front, read each direction once and reuse
    one evaluator. Errors stop evaluation instead of returning partial results.

    Args:
        folder (str or Path): Translation folder, absolute or relative to
            the repository root.
        metric_name (str): 'bleu', 'chrf' or 'comet', case-insensitive.
        metric (object, optional): Evaluator returned by load_metric;
            must match metric_name. If None, load one with default settings.
            Supply this to reuse a model across systems or customize chrF.
        directions (iterable[str]): Direction labels to evaluate. Defaults
            to 'az-en', 'en-az', 'be-en' and 'en-be'. Must not be empty.
        batch_size (int): Sentences per COMET batch. Defaults to 8;
            ignored for BLEU and chrF.
        gpus (int): GPUs for COMET. Defaults to 0 (CPU); use 1 for one
            GPU. Ignored for BLEU and chrF.

    Returns:
        dict[str, dict]: Results keyed by direction. Each result contains
            'direction', 'metric', 'n_sentences' and 'score', plus COMET's
            'sentence_scores' or BLEU/chrF's 'signature'.

    Raises:
        ValueError: If the metric is unsupported, directions is empty,
            input line counts differ, or required files contain no lines.
        FileNotFoundError: If any required translation file is missing.
        ImportError: If a required metric package is unavailable when loading.

    Example:
        results = evaluate_all_directions("out/nllb_basemodel", "bleu")
        english_to_belarusian = results["en-be"]["score"]
    """
    metric_name = _validate_metric(metric_name)
    directions = tuple(directions)

    if not directions:
        raise ValueError("Provide at least one translation direction")

    # Catch missing files before loading COMET or evaluating any direction.
    for direction in directions:
        _translation_paths(
            folder,
            direction,
            include_source=(metric_name == "comet"),
        )

    results = {}

    for direction in directions:
        translations = load_translations(
            folder,
            direction,
            include_source=(metric_name == "comet"),
        )

        # Load once, after checking the first direction's texts.
        # On later iterations, metric already holds the evaluator.
        if metric is None:
            metric = load_metric(metric_name)

        results[direction] = _score_translations(
            translations=translations,
            direction=direction,
            metric_name=metric_name,
            metric=metric,
            batch_size=batch_size,
            gpus=gpus,
        )

    return results
