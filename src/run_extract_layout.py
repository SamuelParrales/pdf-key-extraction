


from common.common_types import LayoutElement
from common.data_storage import DataStorage

from extract.layout_extractor import LayoutExtractor

LINE_TOLERANCE = 3.0  # ajusta entre 2 y 5 según tu caso

def sort_key(e):
    x0, y0, x1, y1 = e["bbox"]

    # “cuantiza” Y para agrupar líneas reales
    y_group = int(y0 / LINE_TOLERANCE)

    return (e["page"], y_group, x0)



def main():
    paths = DataStorage.find_pdf_paths()
    layout_extractor = LayoutExtractor()
    for path in paths:
        try:
            lines, images = layout_extractor.extract_from_path(path)
        except Exception as error:
            print(f"Descartado {path}: no se pudo abrir como PDF ({error})")
            continue

        lines.sort(key=sort_key)
        DataStorage.save_layout_element(filename=path.stem,element=lines)
        DataStorage.save_images(filename=path.stem, images=images)

main()

