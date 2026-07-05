from pathlib import Path

from extract.key_value_extractor import KeyValueExtractor


pdf_path = Path('test_pdf/2905202601131547816200120020010000000010552061816.pdf')

extractor = KeyValueExtractor()
extractor.predict(pdf_path)
