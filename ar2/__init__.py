"""ar2: controller-driven architecture search for 200 nt episomal MPRA models (AutoResearch v2).

Code decides what to explore, an LLM writes one model per assignment, and the product is an archive with the best tuned model of each distinct family
and a complete record of every run. See ar2/CONTRACT.md for the seams between the packages and the specification for the behaviour.
"""

__version__ = "0.1.0"
