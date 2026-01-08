from fabula.segment import ParagraphSegmenter, RegexSentenceSegmenter

def test_paragraph_segmenter():
    seg = ParagraphSegmenter()
    text = "A.\n\nB."
    out = seg.segment(text)
    assert len(out) == 2
    assert out[0].text == "A."
    assert out[1].text == "B."

def test_sentence_segmenter():
    seg = RegexSentenceSegmenter()
    text = "Bonjour. Ça va ? Oui!"
    out = seg.segment(text)
    assert len(out) >= 2

