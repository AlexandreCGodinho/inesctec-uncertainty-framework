Tu es un reviewer sévère. Tu dois exécuter la **self-review checklist** exigée par Álvaro avant la review d'Alexandre. Vérifie chaque point rigoureusement et signale immédiatement tout problème.

## Contexte

Le projet compare 4 patterns agentiques sur des métriques d'incertitude. Le code est sur la branche `p2-pipeline` du repo `inesctec-uncertainty-framework`. Les PRs sont :
- PR#1 (`p2-core → main`) : LLMClient, Telemetry, Scorers
- PR#2 (`p2-pipeline → p2-core`) : A1-A3 (Data Join, AUROC, Pareto)

## Checklist à vérifier

### 1. Composants manquants (dev split)
Le dev split n'a pas de u_P (logprobs absents) et seulement 3 datasets (pas de LiveBench). Vérifie :
- [ ] `build_telemetry()` : que se passe-t-il si `logprobs` est `None` → `compute_u_P(None, u_F)` doit retourner 0.0 sans planter
- [ ] `compute_u_P()` dans `src/metrics/u_P.py` : accepte `None` comme logprobs ?
- [ ] `analysis_auroc.py` : que se passe-t-il si `U_surface`, `U_composite` ou `U_surface_no_uR` sont `NaN` ? Le script les ignore via `dropna()` ?
- [ ] `analysis_pareto.py` : idem, gère les NaN ?
- [ ] `generate_score_table.py` : si un dataset manque, le script ne plante pas (juste pas de fichier dans le dossier) ?
- [ ] Exécute les 3 scripts d'analyse avec des données où u_P = None et vérifie qu'ils ne plantent pas.

### 2. per_item_scores.csv : contenu
- [ ] Exécute `generate_score_table.py` sur les données mock complètes.
- [ ] Vérifie que le CSV contient les **4 variantes IFEval** : `strict_prompt`, `loose_prompt`, `strict_instruction`, `loose_instruction`
- [ ] Vérifie que le CSV contient `match` pour PopQA et `execution_accuracy` pour FinQA
- [ ] Vérifie que `headline_bin` est présent et correct :
  - PopQA : 1 si match = 1.0, 0 sinon
  - IFEval : 1 si strict_prompt = 1.0, 0 sinon
  - FinQA : 1 si execution_accuracy = 1.0, 0 sinon
- [ ] Vérifie qu'il n'y a PAS de colonne générique "accuracy" ou "Accuracy (%)"

### 3. termination_reason
- [ ] Vérifie que `termination_reason` est dans le CSV final (colonne présente, pas vide)
- [ ] Vérifie que la valeur provient bien du record JSONL de télémetry (pas une valeur par défaut)
- [ ] Vérifie dans `generate_score_table.py` : `record.get("termination_reason", "")` ligne 56

### 4. Plots "DEV ONLY" label
- [ ] Vérifie dans `analysis_auroc.py` et `analysis_pareto.py` : existe-t-il un paramètre `--dev` ou un moyen d'ajouter "DEV ONLY" aux titres des plots ?
- [ ] Si non, signale-le comme un problème : Álvaro exige que les plots issus de dev data soient étiquetés.
- [ ] Les scripts doivent accepter un flag `--dev` qui ajoute "(DEV ONLY)" dans les titres des figures.

## Règles projet à appliquer

- Pas de valeurs hardcodées (tout dans `config/experiment.yaml`)
- Reproductibilité : pas de random sans seed fixe
- Conventions de nommage : "pattern" pas "architecture"
- Déterminisme : les mêmes entrées donnent les mêmes sorties

## Comment exécuter la vérification

```powershell
# 1. Générer les données mock
python scripts/generate_mock_telemetry.py

# 2. Scorer
python -m src.scoring.score_telemetry --input telemetry\popqa_all.jsonl --output telemetry\scored\popqa_scored.jsonl --dataset popqa --reference telemetry\references\popqa_references.jsonl
python -m src.scoring.score_telemetry --input telemetry\ifeval_all.jsonl --output telemetry\scored\ifeval_scored.jsonl --dataset ifeval --reference telemetry\references\ifeval_references.jsonl

# 3. Join
python scripts\generate_score_table.py --raw-telemetry-dir telemetry --scored-telemetry-dir telemetry\scored --output telemetry\per_item_scores.csv

# 4. Analyser le CSV (colonnes, valeurs)
python -c "import pandas as pd; df = pd.read_csv('telemetry/per_item_scores.csv'); print('Colonnes:', list(df.columns)); print('termination_reason present:', 'termination_reason' in df.columns); print('IFEval scores:', [c for c in df.columns if 'strict' in c or 'loose' in c]); print('headline_bin sample:', df[['query_id','dataset','headline_bin']].head().to_string())"

# 5. Vérifier que les scripts gèrent les NaN
python -c "import pandas as pd; df = pd.read_csv('telemetry/per_item_scores.csv'); df['u_P'] = None; df['U_surface'] = None; from scripts.analysis_auroc import compute_auroc; print('AUROC with NaN:', compute_auroc(df, 'U_surface'))"
```

## Format de réponse

Pour chaque point de la checklist, réponds :
- ✅ PASS : [justification]
- ❌ FAIL : [description précise du problème avec fichier:ligne]
- ⚠️  WARN : [observation qui n'est pas un bug mais mérite attention]

Si au moins un ❌ FAIL est trouvé, bloque la review jusqu'à correction.
