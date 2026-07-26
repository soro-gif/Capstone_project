from billsum.multipublic import summarize_for_audience

text = (
    "Cette loi établit des dispositions relatives à ce texte législatif. "
    "sont interdits et punis par la loi, l'esclavage, le travail forcé, "
    "les traitements inhumains cruels, dégradants et humiliants, la torture "
    "physique ou morale, les violences physiques et les mutilations et toutes "
    "les formes d'avilissement de l'être humain."
)

result = summarize_for_audience(text, "JURISTE", check_factuality=False)
print("SUMMARY:\n", result.summary)
print("COVERED SECTIONS:\n", result.covered_sections)
print("OMITTED SECTIONS:\n", result.omitted_sections)
