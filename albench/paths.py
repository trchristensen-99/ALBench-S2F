"""Deprecated location -- moved to `albench.core.paths` (see docs/STRUCTURE.md).

This shim ALIASES the module rather than re-exporting from it. A `from ... import *`
shim would copy only public names, so `albench.paths` and `albench.core.paths` would hold
SEPARATE state: patching or mutating a private attribute on one would not be seen by
code reading the other. Binding the same module object into both names makes the old
path genuinely the same module, which is what a transitional shim has to be.

Remove once every caller imports from `albench.core.paths`.
"""

import sys

from albench.core import paths as _module

sys.modules[__name__] = _module
