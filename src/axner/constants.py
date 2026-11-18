"""Project-wide constants.

The canonical label list defines the label id order used everywhere in this
repository (configs, datasets, models, metrics). WikiANN label ids are
remapped to this order at load time if the dataset uses a different order.
"""

# Label id used for tokens that are excluded from training/evaluation
# (special tokens, non-first subtokens, truncation padding).
IGNORE_INDEX = -100

# Entity types supported by WikiANN and this project.
ENTITY_TYPES = ("PER", "LOC", "ORG")

# Canonical IOB2 label list: O first, then B-/I- pairs grouped by entity type.
LABEL_NAMES = (
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
)

LABEL_SCHEMES = ("iob2",)
WIKIANN_DATASET_NAME = "wikiann"

# Research languages (ISO-639-1 codes matching WikiANN configs).
SOURCE_LANGUAGES = ("id", "ms", "tl", "en")
TARGET_LANGUAGES = ("jv", "su")
SUPPORTED_LANGUAGES = SOURCE_LANGUAGES + TARGET_LANGUAGES

SAMPLING_METHODS = ("uniform_no_replacement",)
ISSUE_POLICIES = ("report_only", "drop")
DUPLICATE_POLICIES = ("report_only", "drop_train_dups")
SUBWORD_POLICIES = ("first_subtoken_label",)
DATASET_NAMES = ("wikiann", "synthetic")
TEST_LANGUAGE_MODES = ("all_targets", "train")
VALIDATION_LANGUAGES = ("source", "target")
SELECTION_METRICS = ("target_validation_entity_f1",)
