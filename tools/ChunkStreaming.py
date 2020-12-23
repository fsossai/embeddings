# Some tools to process big files in chunks, with one pass.

import pandas as pd

class ChunkStreaming:
    def __init__(self):
        