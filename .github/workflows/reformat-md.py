import re
import sys

# Patterns to be replaced
PATTERNS = [
    # Convert string inside $$ to math code block
    (re.compile(r"\$\$(.*?)\$\$", re.DOTALL), re.compile(r"```math\1```")),
]

HEADING_ID_PATTERN = re.compile(r"\(#+\)(.*)$")
def reformat_heading_id(id):
    ...

def reformat_file(file_path):
    ...