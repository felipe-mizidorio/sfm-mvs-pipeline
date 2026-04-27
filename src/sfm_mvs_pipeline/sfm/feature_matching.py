import logging
from pathlib import Path

import pycolmap

logger = logging.getLogger(__name__)


def match_features(
    database_path: Path,
    options: dict,
    device: pycolmap.Device = pycolmap.Device.auto,
) -> None:
    if not database_path.exists():
        raise FileNotFoundError(f"COLMAP database not found: {database_path}")

    method = options["method"]
    logger.info("Starting feature matching (method=%s) on '%s'", method, database_path)

    if method == "exhaustive":
        pycolmap.match_exhaustive(database_path=database_path, device=device)

    elif method == "vocab_tree":
        vt_opts = options["vocab_tree"]
        pairing_options = pycolmap.VocabTreePairingOptions()
        pairing_options.vocab_tree_path = vt_opts["vocab_tree_path"]
        pairing_options.num_nearest_neighbors = vt_opts["num_nearest_neighbors"]
        pycolmap.match_vocabtree(
            database_path=database_path,
            pairing_options=pairing_options,
            device=device,
        )

    else:
        raise ValueError(
            f"Unknown matching method '{method}'. Expected 'exhaustive' or 'vocab_tree'."
        )

    logger.info("Feature matching complete: %s", database_path)
