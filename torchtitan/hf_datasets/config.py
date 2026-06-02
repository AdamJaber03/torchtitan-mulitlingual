import os


def load_output_base_dir():
    """
    Load the OUTPUT_BASE_DIR from the .data_paths configuration file.

    This function is used across the codebase to ensure all data loading paths
    point to the same configured base directory. The configuration file should
    be created locally by each developer/user.

    Returns:
        str: The base directory path for local data storage

    Raises:
        FileNotFoundError: If .data_paths configuration file doesn't exist
        ValueError: If OUTPUT_BASE_DIR is not set or is a placeholder path
    """
    # Find .data_paths at the project root (3 levels up from this file)
    config_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        ".data_paths"
    )

    if not os.path.exists(config_file):
        raise FileNotFoundError(
            f"Data paths configuration file not found at {config_file}.\n"
            f"To set up your local data paths, run:\n"
            f"  touch {config_file}\n"
            f"  echo 'OUTPUT_BASE_DIR=<your_local_data_path>' >> {config_file}\n\n"
            f"Or copy the template:\n"
            f"  cp .data_paths.example .data_paths\n"
            f"Then edit .data_paths with your actual data path."
        )

    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('OUTPUT_BASE_DIR='):
                path = line.split('=', 1)[1]
                if path in ['<your_local_data_path>', '/path/to/your/data']:
                    raise ValueError(
                        f"Warning: Placeholder path detected in {config_file}.\n"
                        f"Please edit the file and replace:\n"
                        f"  OUTPUT_BASE_DIR={path}\n"
                        f"with your actual local data path:\n"
                        f"  OUTPUT_BASE_DIR=/your/actual/data/path"
                    )
                return path

    raise ValueError(
        f"OUTPUT_BASE_DIR not found in {config_file}.\n"
        f"Add the following line to the file:\n"
        f"  OUTPUT_BASE_DIR=/your/local/data/path"
    )
