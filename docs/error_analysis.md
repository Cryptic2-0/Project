# Error Analysis — Baseline Model (TF-IDF + Logistic Regression)

**Dataset:** IMDb test split (7 437 reviews)
**Model:** `models/baseline.joblib`
**Date:** 2026-05-24

---

## False Positives (label = negative, predicted = positive)

The model assigned positive sentiment to reviews that are actually negative.

---

### FP-1

> *"i can't believe this is on dvd. even less it was available at my local video store. some argue this is a good movie if you take in consideration it had only a 4000$ budget. i find this funny..."*

**Hypothesis:** The reviewer quotes a counterargument ("is a good movie") verbatim before dismissing it. A bag-of-words model cannot tell quoted text from the reviewer's own voice, so tokens like `good` and `movie` add positive weight.

---

### FP-2

> *"trekkies is really not a movie about star trek fandom. it's a freak show about those star trek fans who have no sense of reality. as a freak show, it's fine..."*

**Hypothesis:** "it's fine" appears mid-review as a grudging concession before the reviewer pivots back to criticism. The model anchors on surface-level positive tokens (`fine`, `nice`) without tracking the adversative structure ("but it is a mistake to think...").

---

### FP-3

> *"the film starts out great, with a mock instruction film about the habits of swedish housewives. after that we get a detailed reconstruction of post-war scandinavia with lots of amazing cars, electronic equipment and interior design; a minimal jazz score, nice cinematography and stylish titling..."*

**Hypothesis:** Classic front-loaded praise followed by a negative turn (truncated at 300 chars). Early positive tokens (`great`, `amazing`, `nice`, `stylish`) dominate the TF-IDF vector; the negative critique appears later and carries less weight in a global bag-of-words representation.

---

### FP-4

> *"nat (voiced by trevor gagnon), along with his brainiac friend iq and the always hungry scooter are kids with big dreams. they want to be the first flies in space. and what encourages their dreams is the first spacecraft to land on the moon, the apollo 11..."*

**Hypothesis:** The reviewer describes the plot — dreams, Apollo 11, space — using language that is inherently positive or aspirational. These plot-descriptive tokens (`dreams`, `encourages`) have no inherent sentiment, but co-occur with positive reviews in training data, causing a false trigger.

---

### FP-5

> *"stan 'n' ollie get mixed up with a couple of floozies after setting out to visit a theatre which burns down in their absence! needless to say, their tyrannical wives..."*

**Hypothesis:** Plot summary of a comedy short film. Neutral / narrative tokens dominate; the review doesn't use explicit negative evaluation language. The model lacks a prior that silent-era comedy synopses are not sentiment-bearing.

---

## False Negatives (label = positive, predicted = negative)

The model assigned negative sentiment to reviews that are actually positive.

---

### FN-1

> *"as a big dostoyevsky fan, i had always been disappointed with hollywood's halfhearted attempts to get into the russian romantic aesthetic — case in point, yul brynner as dmitri karamazov. i had thought the whole problem was a poor casting decisions, but then i saw yul as major surov and changed my..."*

**Hypothesis:** The review opens with a list of prior disappointments (`disappointed`, `halfhearted`, `poor`) before pivoting to praise. The negative opening front-loads strong negative signal; the positive resolution is truncated. The model cannot capture this narrative arc.

---

### FN-2

> *"i am a big fan of cinema verite and saw this movie because i heard how interesting it was. i can honestly say it was very interesting indeed. the two lead actors are awesome, the film isn't ever boring, and the concept behind it (though obviously inspired by the columbine killings and the home movie..."*

**Hypothesis:** The review praises the film while anchoring it to the Columbine massacre. Tokens from the dark subject matter (`killings`, `columbine`) carry strongly negative weight that overwhelms genuine positive terms (`awesome`, `interesting`).

---

### FN-3

> *"a bunch of mostly obnoxious and grossly unappealing teens go to a creepy, remote, rundown old mortuary... to attend an anything-goes all-out halloween party..."*

**Hypothesis:** This is a positive horror review. Genre-specific vocabulary (`obnoxious`, `creepy`, `mortuary`) is used approvingly to describe atmosphere, but these tokens are almost exclusively negative in non-horror reviews. The model has no genre awareness.

---

### FN-4

> *"i enjoyed the acting in this movie. except for the sister... my main problems with the movie were the anticlimatic ending and the execution scene..."*

**Hypothesis:** Balanced review that explicitly lists criticisms under "my main problems". The negative section uses more specific and emphatic language (`problems`, `anticlimatic`, `obnoxious`) than the positive section (`enjoyed`), skewing the TF-IDF weighting toward negative despite overall positive intent.

---

### FN-5

> *"at this point it seems almost unnecessary to state that jon bon jovi delivers a firm, strong, seamless performance as derek bliss. his capability as an actor has been previously established by his critical acclaim garnered in other films..."*

**Hypothesis:** Formal, understated praise. Tokens like `firm`, `seamless`, `previously established` are not high-frequency positive tokens in the training vocabulary. The model relies on tokens like `great`, `excellent`, `loved`; when reviewers use formal register, positive signal is underrepresented.

---

## Summary of Failure Modes

| Mode | FPs | FNs |
|------|-----|-----|
| Quoted / reported speech | FP-1 | — |
| Concession tokens (`fine`, `good`) used ironically | FP-2 | — |
| Front-loaded praise, negative ending | FP-3 | FN-1 |
| Plot-descriptive tokens mistaken for sentiment | FP-4, FP-5 | — |
| Dark subject matter overrides positive intent | — | FN-2 |
| Genre-specific vocabulary (horror) | — | FN-3 |
| Explicit criticism section dominates | — | FN-4 |
| Formal / understated praise vocabulary | — | FN-5 |

**Implications for Day 6+ (DistilBERT):** A contextual model should fix FP-1 (quoted speech), FP-2 (concessive structure), FN-1 (narrative arc), and FN-5 (register). Genre-vocabulary issues (FN-3) may persist without genre-specific fine-tuning data.
