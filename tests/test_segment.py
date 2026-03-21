from fabula.segment import ParagraphSegmenter, RegexSentenceSegmenter


def test_paragraph_segmenter_drops_short_blocks_and_tracks_positions():
    text = "A\n\nThis is a full paragraph.\n\nB"
    segs = ParagraphSegmenter(min_len=5).segment(text)

    assert len(segs) == 1
    assert segs[0].text == "This is a full paragraph."
    assert 0.0 < segs[0].rel_pos < 1.0


def test_regex_sentence_segmenter_splits_on_terminal_punctuation():
    text = "Bonjour! Ça va? Très bien."
    segs = RegexSentenceSegmenter().segment(text)

    assert [s.text for s in segs] == ["Bonjour!", "Ça va?", "Très bien."]
    assert segs[0].idx == 0
    assert segs[-1].idx == 2
