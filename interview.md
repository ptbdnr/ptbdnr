# Outline

1. welcome
2. ask permission to use note-taker
3. present agenda, "3min for context then we go deep into the details"
4. explain interruptions
5. intro (keep self-intro short)
6. motivation and target next position
7. what helps you the best to understand the company and/or org more?
8. previous experience:
  * design alternatives (options, pro-cons), solution alternatives and why? example and counter example “What alternatives did you consider?”
  * claim-evidence-measurement. ask specific: a name (model name), a number (team size, latency time breakdown, data volume, retention period, QPS), or a failure.
  * evaluation: probe (“How did you know it was better?”)
  * ownership: When a candidate stalls on an ownership question, stay on it with simpler, factual sub-questions rather than moving on.
  * learning: what would you do differently?
8. scenario
9. candidate's question

# Style

* set expectations: “I'm going to give you 3min for the context, then we will go deep into the design and delivery.”
* one short question at a time. 
* if answer is unstructured then interrupt:
   * steer “Let's focus on this slightly different way. Let's focus on this as a technical problem.”
   * drill “Can I pick into a couple of details here?”
* if false answer, allow to correct, don't affirm incorrectly:
  * can you be more specific
  * ask process/decision “How do you reason about whether it will scale?” “what if … how would you want to solve it?”\
  * move one: “Ok. Let's leave that one.”

# What-if scenario

to test reasoning. If candidates claim is conflicting then state the tension neutrally, offer a hint and observe recovery.

* stakeholder: client wants it in {few} weeks, it requires {many} weeks
* design: public-sector doc processing, {many} docs, classified as OFFICIAL-SENSITIVE, within UK region
* diagnose: retrieval quality dropped {lot}% after a content migration
* diagnose: RAG is hallucinating in prod
* review code

Trivia: avoid because AI-cheat is easy, it test jargon, if a must then ask same questions, scores: 0 wrong; 1 directionally right; 2 precise; 3 precise with trade-offs.
