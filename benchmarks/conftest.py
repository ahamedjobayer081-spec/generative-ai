import sys
import os

# Add the gemhall source to the Python path
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "gemini",
        "sample-apps",
        "gemini-hallcheck",
        "src",
    ),
)
