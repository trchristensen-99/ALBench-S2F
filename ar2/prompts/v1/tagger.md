Classify the PyTorch model below. Judge from the code only; ignore names, comments and docstrings.
{{taxonomy}}
Choose the family by the mechanism that does most of the integration across the sequence.
A convolution stem followed by attention is transformer_hybrid only if attention blocks hold most
of the depth; one attention layer added to a residual CNN is cnn_residual. Use other:<name> only
if no family fits, and justify it in one line.
OUTPUT: one JSON object with keys family, core_block, uses_attention, uses_recurrence,
normalisation, activation, pooling, positional_info, downsampling_factor, justification.
CODE:
{{code}}
