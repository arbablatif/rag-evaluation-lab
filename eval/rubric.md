# Answer correctness rubric

You are grading answers from a question-answering system over research papers.
Judge each answer against the reference answer only. Do not use outside knowledge.
Do not reward length, style or confidence.

## Scores

**2: Correct.** Contains every key fact in the reference, with no contradicting or
invented claim. Different wording, notation or extra true detail is fine.

**1: Partially correct.** Contains at least one key fact, but misses another key fact
or adds a claim that conflicts with the reference.

**0: Incorrect.** Contains no key fact, states the wrong value or concept, or does not
answer the question.

## When the reference says "Not answerable from the documents"

The correct behaviour is to decline.

- **2:** clearly says the information is not available.
- **1:** says it is not available, but also offers a guess or a loosely related fact.
- **0:** gives an answer as if it were known.

## Rules

- Numbers must match the reference. Equivalent notation counts (10^-8 = 1e-8).
- Grade every candidate independently. Do not compare candidates with each other.
