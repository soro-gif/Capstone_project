import sys
sys.path.insert(0, 'src')
from billsum.multipublic import _extract_document_intro, _generate_single_chunk, _clean_output, _merge_intro_summary
from billsum.multipublic import load_model, get_profile

text = (
    "Cette loi établit des dispositions relatives à ce texte législatif. "
    "sont interdits et punis par la loi, l'esclavage, le travail forcé, "
    "les traitements inhumains cruels, dégradants et humiliants, la torture "
    "physique ou morale, les violences physiques et les mutilations et toutes "
    "les formes d'avilissement de l'être humain."
)
model, tokenizer = load_model(device='cpu')
profile = get_profile('JURISTE')
raw = _generate_single_chunk(model, tokenizer, text, profile, 'cpu')
intro = _extract_document_intro([('Texte entier', text)], profile.name)
merged = _merge_intro_summary(intro, raw)
print('INTRO:', intro)
print('RAW:', raw)
print('MERGED:', merged)
print('---')
print('CLEANED RAW:', _clean_output(raw))
