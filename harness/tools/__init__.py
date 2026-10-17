# Re-export the registry for convenient `from harness.tools import registry`.
from harness.tools.registry import registry

# Import each tool module for its side effects (the @tool decorators run on import).
# Add a new tool category? Add its import here.
from harness.tools import filesystem  # noqa: F401 — imported for side effects

# Same reasoning as filesystem above: importing this module runs its
# @tool-decorated git_status/git_diff/git_log/git_commit/git_checkout/git_branch
# definitions, registering them. Nothing here ever calls `git` by name.
from harness.tools import git  # noqa: F401 — imported for side effects

from harness.tools import bash # noqa: F401

from harness.tools import web_search    # noqa: F401 — imported for side effects

from harness.rag import recall    # noqa: F401 — imported for side effects
