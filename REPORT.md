# Assignment 2 Report

<!--
CÓMO USAR ESTA PLANTILLA
- `___`            → un número. El comentario de al lado dice de qué archivo sale.
- [TU ANÁLISIS: …] → texto tuyo. Las preguntas son una guía; bórralas al escribir.
- m = artifacts/metrics.json · op = artifacts/operating_point.json
- Copia los números DESPUÉS de relanzar el workflow con tus costes definitivos.
- Borra estos comentarios cuando termines (no se ven en GitHub, pero mejor limpio).
-->

- **Name**:
- **Student ID**:
- **Email**:
- **Group**: [BBADBA 5A | BBADBA 5B | PPLEDBA 5A | BDBA 3A]

## Business problem, and the cost of being wrong

**Decision supported:** [TU ANÁLISIS: ¿quién usa el modelo (un recolector, una asociación micológica, una app), en qué momento, y qué hace con la respuesta?]

**Positive class:** Poisonous-mushroom.

| Outcome | What happens in practice | Cost (€ per mushroom) |
|---|---|---|
| True positive (poisonous, thrown away) | [qué pasa] | 0 <!-- costs.json: tp --> |
| False negative (poisonous, eaten) | [qué pasa] | 1,000 <!-- costs.json: fn --> |
| False positive (edible, thrown away) | [qué pasa] | 10 <!-- costs.json: fp --> |
| True negative (edible, eaten) | [qué pasa] | 0 <!-- costs.json: tn --> |
| Sent to a person for review | [quién revisa y cuánto cuesta su tiempo] | 3 <!-- costs.json: review --> |

**Why the two errors cost different amounts:** [TU ANÁLISIS: ¿cuántas veces peor es un FN que un FP y por qué? ¿De dónde sale cada cifra?]

## Dataset

- **Roboflow Universe:** workspace `jamurdetection`, project `jamur-detection`, **version 2**
- **Link:** https://universe.roboflow.com/jamurdetection/jamur-detection/dataset/2
- **Task:** object detection, 2 classes: `Edible-mushroom`, `Poisonous-mushroom`
- **Used as published** (downloaded directly, not forked).
- **What one image represents:** [TU ANÁLISIS: ¿qué se ve en una imagen típica? ¿una seta, varias, en el bosque, de catálogo?]

**Why this dataset serves the problem:** [TU ANÁLISIS: ¿se parecen estas fotos a las que haría un recolector con el móvil? ¿en qué no?]

## What the dataset's composition implies

| Split | Images | Edible instances | Poisonous instances | Median box area (% of image) |
|---|---|---|---|---|
| Train | 694 | 1,553 | 830 | 9.7 % |
| Validation | 98 | 296 | 115 | 7.2 % |
| Test | 50 | 121 | 65 | 8.5 % |
<!-- m → dataset.stats.<split>.images / .instances / .box_area_share_of_image.median (multiplica ×100 para %) -->

- Train images are augmented by Roboflow (2 outputs per source image: flips and crop), so the 694 train images come from about 347 originals. Validation and test are not augmented.
- Box size: 10 % of boxes cover less than 1.9 % of the image, 10 % cover more than 44.9 %. <!-- m → box_area_share_of_image.p10 / .p90 del split test -->

**What this implies:** [TU ANÁLISIS: ¿hay desequilibrio entre clases? ¿la clase positiva tiene suficientes instancias en test para fiarse de su precisión/recall? ¿qué no se puede concluir con solo 50 imágenes de test? ¿se parece el test a producción?]

## Finetuning and baselines

**Where and how I trained:** GitHub Actions (`.github/workflows/train.yml`), CPU-only runner, `yolo11n` from the COCO-pretrained checkpoint, 30 epochs, image size 640, seed 0, all other settings Ultralytics defaults. Training took about 1 h 30 min. <!-- m → training; duración en la pestaña Actions -->

**Why detection with yolo11n:** [TU ANÁLISIS: ¿por qué detección y no clasificación (pista: 2 clases)? ¿por qué el modelo pequeño?]

Results on the same test split:

| Model | mAP50 | mAP50-95 | € per 1,000 mushrooms at the chosen threshold |
|---|---|---|---|
| Finetuned yolo11n | 0.956 | 0.828 | 3,077 |
| COCO yolo11n zero-shot | 0.000 | 0.000 | 200,000 |
| Trivial: predict nothing | 0.000 | 0.000 | 200,000 |
<!-- m → finetuned.map50 / map50_95; op → test_policies.model_alone.cost_per_1000;
     m → baselines.<nombre>.test_cost_at_chosen_threshold.cost_per_1000 -->

The COCO model has no mushroom class, so it cannot score on our classes. It put a box on 19.9 % of the test mushrooms (37 of 186), mostly labelled as donut (18), vase (8) and banana (5). <!-- m → baselines.coco_zero_shot.share_of_mushrooms_boxed_as_any_coco_class / coco_labels_it_used -->

**Interpretation:** [TU ANÁLISIS: ¿cuánto mejora el modelo a los baselines? ¿y en euros? Si en euros no los mejora, explícalo: es un hallazgo válido según el enunciado.]

## Per-class evaluation and failure modes

| Class | Precision | Recall | AP50 | AP50-95 |
|---|---|---|---|---|
| Edible-mushroom | 0.910 | 0.884 | 0.951 | 0.858 |
| Poisonous-mushroom (positive) | 0.941 | 0.877 | 0.961 | 0.798 |
| All (mean) | | | 0.956 | 0.828 |
<!-- m → finetuned.per_class -->

Confusion matrix on the test split (confidence 0.25):

| Predicted ↓ / True → | Edible | Poisonous | Background |
|---|---|---|---|
| Edible | 112 | 0 | 20 |
| Poisonous | 0 | 61 | 9 |
| Background (missed) | 9 | 4 | — |
<!-- m → finetuned.confusion_matrix (filas = predicho, columnas = real). Imagen: artifacts/confusion_matrix.png -->

**Which class carries the headline number:** [TU ANÁLISIS]

**Where the model fails:** [TU ANÁLISIS: ¿cuántas venenosas se llaman comestibles? ¿cuántas se pierden (background)? Abre artifacts/confidences_test.csv, mira las venenosas con poisonous_conf baja y busca esas imágenes: ¿qué tienen en común?]

## Cost matrix and operating point

**Costs:** see the table in the first section. **Base rate assumed:** 20 % of mushrooms brought for checking are poisonous. <!-- costs.json: base_rate --> [TU ANÁLISIS: ¿de dónde sale esa tasa?]

**Unit of decision:** one mushroom. Its score is the model's highest *Poisonous* confidence on a box overlapping it (IoU ≥ 0.5), or 0 if there is none. At threshold *t*, a mushroom fails (thrown away) if its score is ≥ *t* and passes (eaten) otherwise. Costs per 1,000 mushrooms are rates within each true class, reweighted to the assumed base rate.

**How the threshold was chosen:** swept *t* from 0 to 1 in steps of 0.01 on the **validation** split and took the minimum; the cost reported below is on the **test** split, so it is not tuned on the data it is reported on.

![Cost against threshold](artifacts/cost_vs_threshold.png)

| | Value |
|---|---|
| Chosen threshold | 0.01 <!-- op → single_threshold --> |
| Test cost at that threshold (€ per 1,000) | 3,077 <!-- op → test_policies.model_alone.cost_per_1000 --> |
| Test FN / FP at that threshold | 1 of 65 poisonous / 0 of 121 edible <!-- op → test_policies.model_alone.fn / .fp --> |

**Assumption that would most change the answer:** [TU ANÁLISIS: formato "Asumí X porque Y, y la respuesta cambia si X baja de Z". Prueba en el dashboard qué cifra hace moverse el umbral.]

## What a person still has to check

| | Value |
|---|---|
| Lower threshold (PASS below) | 0.00 <!-- op → review_band.t_low --> |
| Upper threshold (FAIL at or above) | 0.02 <!-- op → review_band.t_high --> |
| Share sent to review | 80.9 % <!-- op → test_policies.model_plus_review.review_share --> |
| Mushrooms checked per day (assumed) | 500 <!-- costs.json: cases_per_day --> |
| Reviews per day | 405 <!-- op → test_policies.model_plus_review.reviews_per_day --> |
| Minutes per review (assumed) | 2 <!-- costs.json: review_minutes --> |
| Reviewers needed (8 h/day each) | 1.69 <!-- op → .reviewers_needed; costs.json: reviewer_hours_per_day --> |

**Is that realistic?** [TU ANÁLISIS: ¿existe esa gente (micólogos, farmacéuticos)? ¿a qué hora llegan las setas?]

Priced against the alternatives (test split, € per 1,000 mushrooms):

| Policy | € per 1,000 | Share reviewed | Reviewers needed |
|---|---|---|---|
| Do nothing (everything eaten) | 200,000 | 0 % | 0 |
| A person checks everything | 3,000 | 100 % | 2.08 |
| Model decides alone (t = 0.01) | 3,077 | 0 % | 0 |
| **Model + human review (0.00 – 0.02)** | **2,428** | 80.9 % | 1.69 |
<!-- op → test_policies.do_nothing / person_checks_all / model_alone / model_plus_review -->

**Recommendation:** [TU ANÁLISIS: ¿qué opción desplegarías y por qué? ¿el modelo es lo bastante bueno para usarse?]

## Limitations & next steps

[TU ANÁLISIS: limitaciones reales que hayas encontrado (tamaño del test, fotos poco parecidas a las de un recolector, especies que no aparecen, si las etiquetas son fiables…) y, para cada una, qué harías con más tiempo o datos.]

## Generative AI use disclosure

<!-- Revisa y ajusta este texto para que describa exactamente lo que hiciste. -->
I used Claude (Anthropic) through Claude Code to suggest candidate datasets, to write the code (`train.py`, `costing.py`, `app.py`, `.github/workflows/train.yml`), and to format this report template. The problem framing, the cost figures and base rate, the choice of thresholds and review capacity, and all analysis and conclusions in this report are my own.
