# Dataset
Utilities to standardise software development in the AI4BMR group.

# Installation

```bash
pip install "git+https://github.com/AI4SCR/ai4bmr-core.git@main#egg=ai4bmr-core"
# for private repositories
pip install "git+ssh://git@github.com/AI4SCR/ai4bmr-core.git@main#egg=ai4bmr-core"

# or for development
git clone git+ssh://git@github.com/AI4SCR/ai4bmr-core.git
cd ai4bmr-core
pip install -e ".[dev, test]"
pre-commit install
```

# ToDo
- Where should we keep the graph builder? It adds the whole scikit-learn as a dependency and it does not feel like the right place here
