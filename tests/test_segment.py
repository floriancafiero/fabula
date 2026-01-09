from fabula.segment import DocumentChunkTokenSegmenter, ParagraphSegmenter, RegexSentenceSegmenter


class DummyTokenizer:
    def __call__(self, text, add_special_tokens=False, return_attention_mask=False, return_offsets_mapping=False,
                 return_tensors=None):
        tokens = text.split()
        offsets = []
        cursor = 0
        for token in tokens:
            start = text.index(token, cursor)
            end = start + len(token)
            offsets.append((start, end))
            cursor = end
        data = {"input_ids": list(range(len(tokens)))}
        if return_offsets_mapping:
            data["offset_mapping"] = offsets
        return data

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=True):
        return " ".join([f"tok{idx}" for idx in ids])

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


def test_document_chunk_segmenter_offsets():
    tokenizer = DummyTokenizer()
    seg = DocumentChunkTokenSegmenter(tokenizer=tokenizer, chunk_tokens=3, stride_tokens=3, min_tokens=1)
    text = "one two three four five six"
    out = seg.segment(text)
    assert len(out) == 2
    assert out[0].start_char == 0
    assert out[0].end_char == len("one two three")
    assert out[1].start_char == len("one two three") + 1
