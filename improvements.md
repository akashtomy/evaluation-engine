### Two changes before the next release

**1. Introduce an AI Gateway**

Put an AI Gateway between agents and LLM providers to centralize:

* Pre-call policy checks, RBAC, PII detection, quotas and model routing
* Post-call validation, masking/redaction and blocking
* Audit, cost and usage tracking

This provides a consistent guardrail layer instead of implementing policies independently in every agent.

**2. Expand the golden dataset and strengthen evaluation**

Increase the ground-truth/golden traces, especially around failure boundaries such as tool correctness, authorization, ambiguity, privacy, temporal reasoning and incomplete evidence.

Complement the existing deterministic + LLM checker with **DeepEval** metrics such as Task Completion, Tool Correctness, Argument Correctness, Step Efficiency and Plan Adherence, and use **G-Eval** for rubric-based semantic evaluation.

### Hybrid checker trade-off

| Approach             | Cost        | Latency     | Accuracy / Coverage                      |
| -------------------- | ----------- | ----------- | ---------------------------------------- |
| Deterministic checks | Low         | Very Low    | High for known/mechanical rules          |
| DeepEval             | Medium      | Medium      | Good for measurable agent behavior       |
| G-Eval / LLM Judge   | Medium–High | Medium–High | High semantic coverage, some variability |
| **Hybrid**           | **Medium**  | **Medium**  | **Best overall balance**                 |

The key is to use deterministic checks for **hard safety/privacy/authorization gates** and reserve LLM-based evaluation for cases requiring semantic reasoning.
