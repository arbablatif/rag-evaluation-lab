# LEARNING.md: How this works, in plain language

## 1. What this project is for

RAG means *look it up first, then answer*. A retriever finds passages from your documents, and a language model answers using only those passages.

Building a RAG system takes many small decisions. This lab tests three of them on the same documents and questions, so that any difference comes from the decision and not from something else.

> Client version: "Before tuning your system by feel, we measure which settings actually change the answers, and how sure we can be."

## 2. The three design choices

**Chunk size** is how much text goes into each searchable piece.
- Small chunks are precise, but an answer spread over several paragraphs can get split apart.
- Large chunks keep context together, but they send more text to the model and their meaning is more blended.

A **token** is roughly three-quarters of a word. Here, chunks of 256, 512 and 1024 tokens run from about half a page to more than a page.

**The embedding model** turns text into numbers, so that similar meanings land close together. Both models here run locally, so documents stay on your machine and searches are free.

**Top-k** is how many chunks are handed to the LLM. A higher k gives more chances to include the right passage, at a higher cost per question.

## 3. The hidden trap: truncation

This is the most useful lesson in the project. Every embedding model has a maximum input length, and it **silently ignores anything longer**. No error appears.

MiniLM reads 256 tokens. Give it a 1024-token chunk and it builds that chunk's search vector from the first quarter of the text. If the answer sits later in the chunk, search can't find it, even though the text is stored.

We measured it: MiniLM never reads 73% of the text in 1024-token chunks. Granite reads up to 8,192 tokens, so it reads everything.

> Analogy: a librarian who files every book after reading only the first chapter.

This is why the experiment uses Granite rather than the more common `bge-small`, which stops at 512. With `bge-small`, the 1024-token chunks would have been truncated by both models.

## 4. How we measure quality

**Hit rate** asks whether retrieval found the evidence. We stored short verbatim snippets for each question and checked for them in the retrieved chunks. This is stricter than matching a page, because a 1024-token chunk covers much more of a page than a 256-token chunk does.

**Answer correctness** asks whether the final answer is right. An LLM can recognise that "10^-8" and "1e-8" mean the same thing, which exact text matching can't. So an LLM acts as judge, using a written rubric: 2 for correct, 1 for partial, 0 for wrong.

A judge can be wrong too, so we protected it in four ways:
- **Written rubric:** the same rules apply to every answer.
- **Blind grading:** it never knows which configuration wrote an answer.
- **Reference answers:** it grades against a known answer, not its own opinion.
- **An independent audit:** a simple fact-matching check ran alongside it. They agreed 94.4% of the time, and I read every disagreement by hand.

**Latency** is reported in two parts, because retrieval (milliseconds, on your machine) and generation (seconds, over the network) behave very differently.

## 5. Why the results come with intervals

With 16 answerable questions, one question moves a score by about 6 points. So a gap of 5 points between two settings could easily be luck.

**A bootstrap interval** estimates that luck. We re-sample the questions thousands of times and see how much the score moves. A result of "94% (81%–100%)" means the true score could plausibly be anywhere in that range.

**A paired comparison** is stronger, because it compares two settings on the same questions. That removes question difficulty from the picture. If the interval for the difference includes zero, we can't say which setting is better.

Only one effect here clearly cleared zero: top-5 finds evidence more often than top-3. The other trends are worth acting on as sensible defaults, but they aren't proven.

> If a client asks "so which one won?", the honest answer is: "1024 tokens with Granite scored best and the direction is consistent, but we'd want more questions to be sure of the exact ranking."

## 6. What we learned

- **Long chunks need a long-context embedder.** Granite's lead grew with chunk size, from 0 to 17 points.
- **The system refuses rather than invents.** No real answer scored 0, and all unanswerable questions were declined. For most businesses, that's the safer way to fail, but too many refusals still frustrate users.
- **Retrieval decides the outcome.** Answers were fully correct 93% of the time after a hit and 11% after a miss.
- **More context barely slowed answers.** Going from about 900 to 5,900 input tokens didn't noticeably change response time, but it does cost more.
- **A hit doesn't guarantee an answer.** On q12, the evidence was found every time, but the model still refused 5 times out of 12.

## 7. Keeping API use low

The project made 251 API calls, down from an initial estimate of 480:
- **Disk cache:** each prompt is sent once. Rerunning costs nothing, and an interrupted run resumes where it stopped.
- **Canonical passage order:** chunks are sorted in document order, so two configurations that retrieve the same chunks produce the same prompt.
- **Batched judging:** one call grades all of a question's answers, which took judging from 240 calls to 16.
- **Rules before the LLM:** an exact refusal has a known score, so it needs no judge.
- **Free work first:** retrieval is local, so problems were fixed before any money was spent.
- **Hard cap and throttle:** a counter on disk stops the run at 300 attempts, and calls are spaced out.
- **`--mock` mode:** the whole pipeline runs with no API calls.

The trade-off with batched judging is that the judge sees answers side by side, which could influence its scores. Separate calls would be cleaner, but would have cost about 125 more calls.

## 8. Mistakes this project caught

Checking your own measurements is part of the work. Each of these would have produced a misleading result:

| Problem | How it showed up | Fix |
|---|---|---|
| Evidence snippets too narrow | A 0% hit rate on a question with several correct passages | Listed all qualifying passages across the corpus |
| Judge references too demanding | "AdaGrad" lost marks on a "which optimizer?" question | Tightened two references and re-judged |
| Misleading truncation metric | "98% truncated" at 256 tokens, when only 2 tokens were lost | Measured the share of text actually unread |
| Granite in slow precision | A warning in the logs, and 294 seconds to index | Forced float32 |
| Noisy latency | The same setup measured 12 ms, then 66 ms | Timed all configurations in shuffled rounds |
| PDF ligatures | "fi" stored as one special character broke text matching | Normalised the text before matching |

> Client version: "A surprising number is a reason to check the measurement before believing the result."

## 9. Design decisions

- **Chunks cross page boundaries.** A page holds roughly 700–1000 tokens, so stopping at page ends would mean 1024-token chunks never reached their size. The cost is that a chunk can't cite a single exact page, which is fine for an experiment but matters for a product that needs citations.
- **NumPy instead of a vector database.** With under 300 chunks, exact search is instant, and it keeps database behaviour out of the comparison. A production system with millions of chunks would need one.
- **Passages in document order.** Models can pay more attention to some positions, so a fixed rule keeps that constant. A production system might put the best match first.
- **Cheapest Flash model.** It's cheap, and a smaller model makes retrieval differences easier to see. A stronger model might score higher and refuse differently.

## 10. Questions a client might ask

**"Should we just copy these settings?"** Use them as a starting point. The ranking could change with your documents, so the valuable part is the method, which can run on your data.

**"Why not just send the whole document?"** Cost and focus. Sending everything for each question gets expensive, and it gives the model more irrelevant text to sort through. It's worth testing for small document sets, though.

**"How many questions do we need?"** For firm conclusions about gaps of 5–10 points, expect 100 or more. Twenty is enough to find large effects and broken pipelines cheaply.

**"What would you try next?"** Reducing refusals. Adding a reranker and softening the refusal prompt both target the largest failure type.
