from pathlib import Path

from extract.key_value_extractor import KeyValueExtractor


pdf_path = Path('evaluate/pdfs/1106202601139174848500120080200000487350004873513.pdf')

extractor = KeyValueExtractor()
result = extractor.predict(pdf_path)

print(result)