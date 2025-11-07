"""Script to create sample PDF files for testing.

Run this script to generate sample PDF files in tests/data/ directory.
"""

import sys
from pathlib import Path
from pypdf import PdfWriter

# Add parent directory to path to import app modules if needed
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def create_sample_pdf(filename: str, pages: int = 1) -> Path:
    """Create a sample PDF file.
    
    Args:
        filename: Name of the PDF file to create.
        pages: Number of pages in the PDF.
        
    Returns:
        Path to created PDF file.
    """
    data_dir = Path(__file__).parent
    pdf_path = data_dir / filename
    
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    
    with open(pdf_path, "wb") as f:
        writer.write(f)
    
    print(f"Created: {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    # Create sample PDFs for testing
    create_sample_pdf("sample_document.pdf", pages=1)
    create_sample_pdf("sample_multi_page.pdf", pages=3)
    create_sample_pdf("test_rag_content.pdf", pages=2)
    
    print("\nSample PDFs created successfully in tests/data/")

