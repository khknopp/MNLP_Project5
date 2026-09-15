from datasets import load_dataset

def get_dataset(dataset_name="neulab/ted_multi"):
    """Loads a HF dataset and returns train, val and test splits."""
    return load_dataset(dataset_name, revision="refs/convert/parquet", split=["train", "validation", "test"])

def get_language_data(dataset, lang_1, lang_2):
    """Returns a dataset filtered by language."""
    pairs = []
    for line in dataset:
        langs = dict(zip(line["translations"]["language"], line["translations"]["translation"]))
        if lang_1 in langs and lang_2 in langs:
            pairs.append((langs[lang_1], langs[lang_2]))
    return pairs