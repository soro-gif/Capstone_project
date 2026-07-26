import sys
sys.path.insert(0, 'src')
# pyrefly: ignore [missing-import]
from billsum.multipublic import _normalize_words, _find_overlap_suffix_prefix, _drop_first_n_words
intro = "Cette loi établit des dispositions relatives à ce texte législatif."
summary = "loi établit des dispositions relatives à ce texte législatif. interdits et punis par la loi..."
print('INTRO WORDS:', _normalize_words(intro))
print('SUMMARY WORDS:', _normalize_words(summary))
print('OVERLAP:', _find_overlap_suffix_prefix(intro, summary, min_words=3))
print('DROP:', _drop_first_n_words(summary, 11))
