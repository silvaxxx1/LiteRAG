# data/data_pipeline.py

from .loader import download_pdf
from .reader import extract_pdf_metadata, read_pdf_pages
from .splitter import spacy_sentencize, chunk_sentences, build_chunks, save_chunks_to_csv
from .analytic import plot_chunk_stats
from config import CHUNK_SIZE, CHUNKS_CSV_PATH
import logging

def process_pdf_pipeline(url, save_path, chunk_size=CHUNK_SIZE, output_csv=CHUNKS_CSV_PATH, show_stats=False):
    download_pdf(url, save_path)

    metadata = extract_pdf_metadata(save_path)
    logging.info(f"PDF Title: {metadata.get('title', 'No title')}")

    pages = read_pdf_pages(save_path)
    pages = spacy_sentencize(pages)
    pages = chunk_sentences(pages, chunk_size)
    chunks = build_chunks(pages)
    save_chunks_to_csv(chunks, output_csv)

    if show_stats:
        plot_chunk_stats(chunks)

if __name__ == "__main__":
    import os
    from config import ASSETS_DIR

    url = "http://alvarestech.com/temp/deep/Deep%20Learning%20by%20Ian%20Goodfellow,%20Yoshua%20Bengio,%20Aaron%20Courville%20(z-lib.org).pdf"
    save_path = os.path.join(ASSETS_DIR, "raw.pdf")

    process_pdf_pipeline(url, save_path, show_stats=True)


