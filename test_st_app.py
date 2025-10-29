import pytest
from st_app import PDFProcessor

class DummyPDF:
    def __init__(self):
        self.pages = [self]
    def extract_text(self):
        return "Alice and Bob worked on the document with Charlie and Dan."

@pytest.fixture
def dummy_pdf_file(tmp_path):
    return DummyPDF()

def test_extract_connections(monkeypatch, dummy_pdf_file):
    import st_app
    monkeypatch.setattr(st_app.pdfplumber, "open", lambda _: dummy_pdf_file)
    processor = PDFProcessor("fake_path.pdf")
    connections = processor.extract_connections()
    # We expect at least a few connections between Alice, Bob, Charlie, and Dan
    assert connections is not None
    assert any(("Alice", "Bob") in conn or ("Alice", "Charlie") in conn for conn in connections)
