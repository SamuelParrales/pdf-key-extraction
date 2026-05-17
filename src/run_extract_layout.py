


from common.data_storage import DataStorage

from extract.layout_extractor import LayoutExtractor



def main():
    paths = DataStorage.find_pdf_paths()
    layout_extractor = LayoutExtractor()
    for path in paths:
        lines = layout_extractor.extract_from_path(path)
        DataStorage.save_layout_element(filename=path.stem,element=lines)

main()

