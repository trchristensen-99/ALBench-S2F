import sys

import haiku as hk
import jax
import jax.numpy as jnp

sys.path.append(".")
from models.alphagenome_heads import MLP512512Head


class DummyEmbeddings:
    def __init__(self, x):
        self.encoder_output = x


def forward(x):
    head = MLP512512Head(name="test", num_tracks=1, output_metadata=None, organism_to_outputs=None)
    return head.predict(DummyEmbeddings(x), 0)


x = jnp.ones((2, 384 // 128, 1536))
forward_t = hk.transform(forward)
rng = jax.random.PRNGKey(42)
params = forward_t.init(rng, x)
out = forward_t.apply(params, rng, x)
print("Output shape:", out.shape)
