# Générateur de résumés législatifs multi-publics — BillSum

Fine-tuning LoRA pour la **summarization contrôlée** de textes législatifs, avec
contrôle des hallucinations et couverture des obligations légales.
Un même projet de loi → résumés adaptés à plusieurs publics (juriste, dirigeant, citoyen).

**Baobab Labs**

## Structure

```
billsum_project/
├── requirements.txt
├── README.md
├── src/billsum/
│   ├── config.py            # tous les hyperparamètres
│   ├── data.py              # splits officiels + tokenisation longue
│   ├── model.py             # modèle seq2seq + LoRA
│   ├── train.py             # entraînement + early stopping (ROUGE-L)
│   ├── generate.py          # génération (beam search)
│   └── metrics/
│       ├── __init__.py      # ROUGE-L, BERTScore, orchestration
│       ├── obligations.py   # couverture des obligations (custom)
│       └── hallucination.py # hallucination rate (NLI)
└── scripts/
    ├── run_train.py         # entraînement de bout en bout
    ├── run_eval.py          # évaluation (test + ca_test)
    └── distill_multipublic.py  # Phase 2 : distillation multi-publics
```

## Installation

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

## Utilisation

Entraînement (splits officiels, train sous-échantillonné à 12k) :

```bash
python scripts/run_train.py
python scripts/run_train.py --train-subset 15000 --epochs 4
```

Évaluation sur `test` (Congress) et `ca_test` (Californie, hors-domaine) :

```bash
python scripts/run_eval.py --adapter ./billsum-lora --n-eval 100
```

## Les 4 métriques

| Métrique | Mesure |
|----------|--------|
| ROUGE-L | recouvrement lexical |
| BERTScore | similarité sémantique |
| **Couverture des obligations** | % d'obligations légales (`shall`/délais/montants/sanctions) retrouvées sémantiquement dans le résumé |
| **Hallucination rate** | % de phrases du résumé non impliquées (NLI) par la source |

## Choix de conception

- **Contexte long** (Long-T5, 4096 tokens) : les obligations se dispersent dans
  tout le texte ; tronquer à 1024 détruit la couverture.
- **Régularisation** : label smoothing (0.1), weight decay, dropout LoRA, early
  stopping sur ROUGE-L val.
- **Généralisation** : `ca_test` évalué séparément (Congress → Californie).

## Phase 2 — multi-publics

1. `python scripts/distill_multipublic.py --n 2000 --out distilled.jsonl`
   (génère les registres DIRIGEANT/CITOYEN via API Claude).
2. `cfg.USE_CONTROL_TOKEN = True`, réentraîner sur le dataset augmenté.
3. Réévaluer chaque registre séparément.

## Version FR — modèle pré-entraîné (sans réentraînement)

Utilise directement le modèle fine-tuné
[sorolamoussa/t5-small-billsum-fr](https://huggingface.co/sorolamoussa/t5-small-billsum-fr)
pour produire un **résumé exécutif** (juriste) et un **résumé citoyen** d'un
projet de loi français, avec traçabilité des sections et détection
d'hallucinations.

```bash
python scripts/run_multipublic_fr.py --file projet_de_loi.txt
python scripts/run_multipublic_fr.py --file projet_de_loi.txt --audiences JURISTE DIRIGEANT CITOYEN
```

Composants (`src/billsum/`) :
- `audience.py` — contrôleur de style : longueur/niveau de détail par public
  (`JURISTE`, `DIRIGEANT`, `CITOYEN`), sans réentraînement.
- `glossary.py` — glossaire juridique FR → langage clair, appliqué au registre CITOYEN.
- `sections.py` — découpage du texte en articles/titres pour la traçabilité.
- `multipublic.py` — orchestration : génération, couverture des sections
  (sections omises signalées explicitement), couverture des obligations
  (patterns FR/OHADA), et hallucination rate (NLI multilingue).

Le rapport signale toujours les sections et obligations **non retrouvées**
dans le résumé — aucune interprétation juridique non sourcée n'est ajoutée.

## Adaptation OHADA / francophone

Pour le marché ivoirien : modifier `OBLIGATION_PATTERNS` dans
`metrics/obligations.py` (verbes déontiques FR, délais, montants FCFA) et passer
l'embedder + le modèle NLI en multilingue.

## Données

BillSum — https://github.com/FiscalNote/BillSum
Textes de loi US (domaine public). Vérifier la licence pour tout usage commercial.
## Lancer l'app avec streamlit
Pour lancer l'app exécutez la commande
streamlit run streamlit_app.py
