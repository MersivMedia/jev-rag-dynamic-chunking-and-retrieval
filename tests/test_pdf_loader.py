import pytest

pymupdf = pytest.importorskip("pymupdf")

from jev_retrieval.parse import load_file  # noqa: E402


def _make_pdf(path):
    doc = pymupdf.open()
    body = ("Retrieval systems split long documents into passages before embed-\n"
            "ding them. The passage boundaries decide what the retriever can find.\n"
            "Poor boundaries separate a claim from the evidence that supports it.")
    for i in range(4):
        page = doc.new_page()
        page.insert_text((72, 40), "Journal of Retrieval Studies, Vol. 3", fontsize=8)  # running header
        page.insert_text((72, 90), f"Section {i + 1} Results", fontsize=16, fontname="hebo")  # bold, larger
        page.insert_text((72, 130), body, fontsize=10)
        page.insert_text((300, 800), str(i + 1), fontsize=9)  # page number
    doc.set_metadata({"title": ""})
    doc.save(str(path))


def test_pdf_loader_rebuilds_paragraphs_and_headings(tmp_path):
    p = tmp_path / "paper.pdf"
    _make_pdf(p)
    d = load_file(str(p))
    t = d.text
    assert d.format == "markdown"
    assert "Journal of Retrieval Studies" not in t, "running header should be removed"
    assert "\n\n1\n\n" not in t and not t.rstrip().endswith("\n4"), "page numbers should be removed"
    assert "# Section 1 Results" in t, "large bold line should become a heading"
    assert "embedding them" in t and "embed- ding" not in t, "line-break hyphen should be joined"
    # hard-wrapped lines of one block come back as one paragraph
    assert "before embedding them. The passage boundaries decide" in t
    assert d.title == "Section 1 Results"
