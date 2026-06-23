"""Wrapper that patches KNOWN_METRIC_BOUNDS before running the EEE lm_eval converter."""
import sys

# Patch in missing metric bounds before any converter import
import every_eval_ever.converters.lm_eval.utils as eee_utils
eee_utils.KNOWN_METRIC_BOUNDS.update({
    "acc_bytes": (0.0, 1.0),          # byte-normalised accuracy (global-PIQA)
    "eclektic_transfer": (0.0, 1.0),  # ECLeKTic weighted recall
    "eclektic_overall": (0.0, 1.0),   # ECLeKTic simple mean recall
})

# Now invoke the CLI entry point as if called from the command line
from every_eval_ever.__main__ import main
sys.exit(main())
