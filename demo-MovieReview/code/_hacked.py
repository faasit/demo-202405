import pathlib
import os
import sys

code_dir = pathlib.Path(__file__).parent.resolve()
sys.path.append(code_dir.as_posix())
sys.path.append((code_dir / "movie").as_posix())
