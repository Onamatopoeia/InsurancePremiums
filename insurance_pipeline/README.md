# Insurance Premium Risk Assessment Pipeline

A machine learning pipeline that predicts medical conditions for insurance applicants, scores their risk, and generates personalised intervention recommendations — with a feedback loop that retrains the model as users complete their interventions.

---

## What it does

When someone applies for medical insurance, the insurer needs to estimate how likely that person is to make claims. Traditionally this is done with broad actuarial tables — age brackets, smoker/non-smoker, that sort of thing.

This pipeline takes a more granular approach:

1. It looks at a user's full health and lifestyle profile
2. Predicts what medical conditions they're likely to develop
3. Scores their insurance risk based on those predictions plus their raw health markers
4. Recommends specific interventions (programs, screenings, lifestyle changes) that would reduce that risk
5. Tracks whether the user follows through, and uses that data to improve future predictions

The end result is a risk score and a personalised action plan for each user, rather than a one-size-fits-all premium bracket.

---

## Project structure

```
insurance_pipeline/
├── main.py                 # Entry point — runs the full pipeline
├── data_generator.py       # Generates synthetic user records for training
├── ml_module.py            # Trains and runs the condition prediction model
├── risk_module.py          # Scores each user's insurance risk (0–100)
├── intervention_module.py  # Produces personalised recommendations per user
├── monitor.py              # Tracks intervention adherence, retrains the model
└── requirements.txt
```

---

## Setup

**Requirements:** Python 3.9+

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python main.py
```

---

## How each module works

### `data_generator.py`
Generates synthetic user records. Each record contains 16 features across four categories:

| Category | Features |
|---|---|
| Clinical | BMI, blood pressure, cholesterol, blood glucose |
| Lifestyle | Smoking status, alcohol intake, exercise frequency, sleep, stress |
| Demographics | Age, employment status, income bracket |
| Insurance history | Prior claims, family history of heart disease / diabetes |

The generator skews condition probabilities based on feature combinations — e.g. users with high BMI and elevated blood glucose are more likely to be assigned Type 2 Diabetes as their condition label. This makes the training data behave realistically rather than randomly.

In production, this module is replaced by a connector to your actual data sources (EHR systems, wearables, insurance records).

---

### `ml_module.py`
Trains a Random Forest classifier to predict which of 8 medical conditions a user is most likely to develop:

- Healthy
- Type 2 Diabetes
- Hypertension
- Cardiovascular Disease
- Obesity
- Asthma
- Depression/Anxiety
- Chronic Back Pain

The model trains on 80% of the generated data and validates on the remaining 20%. With synthetic data the accuracy sits around 26–28% — this is expected, because the data has intentional randomness built in. With real patient data and proper feature engineering, this number climbs significantly.

The module outputs the trained model, the feature list, and a label encoder. These are passed forward to the prediction and risk steps.

---

### `risk_module.py`
Takes a user's raw features and their predicted condition, and produces a risk score from 0 to 100.

The score starts with a base value tied to the predicted condition (Cardiovascular Disease starts at 60, Healthy starts at 5, etc.) and then adds points for specific risk factors:

| Factor | Points added |
|---|---|
| Active smoker | +12 |
| Severely elevated BP (≥160) | +10 |
| Severely obese (BMI ≥35) | +10 |
| Sedentary (≤1 exercise day/week) | +7 |
| High blood glucose (≥126 mg/dL) | +6 |
| High alcohol intake (≥14 units/week) | +6 |
| 5+ prior claims | +8 |
| Age 65+ | +8 |

The final score maps to a risk level:

| Score | Level |
|---|---|
| 0–24 | Low |
| 25–49 | Medium |
| 50–74 | High |
| 75–100 | Critical |

The module also returns the top contributing factors — these are used by the intervention module to decide what to recommend.

---

### `intervention_module.py`
Generates up to 6 personalised recommendations per user. The recommendations are driven by the user's specific risk factors, not their risk level alone.

A 45-year-old smoker with borderline blood pressure gets a smoking cessation program and a home BP monitoring kit. A 65-year-old with Hypertension gets the DASH diet program and age-appropriate cancer screenings. A Critical-risk user gets a care navigator assigned within 48 hours as the first item.

All users on Medium risk or above are shown a Premium Reduction Pathway — completing 3+ interventions within 6 months qualifies them for up to a 15% annual premium discount. This is the mechanism that aligns the user's incentives with the insurer's.

---

### `monitor.py`
Two responsibilities:

**Activity tracking** — simulates user engagement with their assigned interventions. Adherence rates are modelled by risk level (Low-risk users tend to follow through more than Critical-risk users). Each user gets a follow-up risk score reflecting how much their risk dropped based on what they completed.

**Model retraining** — users who showed high adherence (≥70%) and measurable risk reduction are used to augment the training dataset. Their records are upsampled and the model is retrained, so future predictions account for the fact that lifestyle interventions genuinely shift health outcomes.

In production, the activity data would come from app check-ins, wearable device syncs, and health provider reports rather than simulation.

---

## Example output

```
[1/5] Generating dynamic dataset...
      2000 training records | 17 features

[2/5] Training machine learning model...
      Validation accuracy: 28.00%

[3/5] Predicting medical conditions for new users...
      Processed 5 new user(s)

[4/5] Running risk assessment & generating interventions...

  ┌─ USR002  Age: 65  Condition: Hypertension
  │  Risk: [████████████░░░░░░░░] 60/100  (High)
  │  Interventions:
  │    • Priority Health Review — physician within 2 weeks (co-pay waived)
  │    • Hypertension Management Program — DASH diet, BP kit, medication review
  │    • Physical Activity Initiative — 8-week starter plan, gym subsidy
  │    • Preventive Screenings Package — cancer screenings, bone density scan
  │    • Premium Reduction Pathway — 15% discount available
  └───────────────────────────────────────────────────────

[5/5] Simulating user activity monitoring & model retraining...
      USR002: 1/6 interventions completed (17% adherence) → risk score 60 → 59
      Retrained on 2003 records (3 activity-derived samples added)
```

---

## Extending the pipeline

The five modules are deliberately independent. Swapping any one of them out doesn't require touching the others.

**Plug in real data** — replace `data_generator.py` with a loader that reads from your EHR system or a CSV export. The rest of the pipeline expects a pandas DataFrame with the same 16 column names.

**Swap the model** — `ml_module.py` uses Random Forest by default. The `train_model()` function can be modified to use XGBoost, a neural network, or any sklearn-compatible classifier without changing anything downstream.

**Add explainability** — the risk module already surfaces the top contributing factors per user. Adding SHAP values to the ML module would give you feature-level explanations for each condition prediction as well.

**Wrap it in an API** — `main.py` calls each module in sequence. Each function has clean inputs and outputs, so wrapping the pipeline in a FastAPI endpoint is straightforward.
