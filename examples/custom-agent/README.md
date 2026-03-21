# Custom Framework Agent Example

Demonstrates how to implement a custom `FrameworkAdapter` for the Corail runtime.

## Implementing a Custom Adapter

```python
from corail.adapters.frameworks.base import FrameworkAdapter
from corail.adapters.llms.base import LLMAdapter
from corail.core.agent_config import AgentConfig


class MyCustomAdapter:
    def supports(self, framework: str) -> bool:
        return framework == "my-custom"

    async def execute(self, config: AgentConfig, input_text: str, llm: LLMAdapter) -> str:
        # Custom logic here
        messages = [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": input_text},
        ]
        return await llm.generate(messages, config.model, config.temperature)
```

Then register it:

```python
from corail.adapters.factory import AdapterRegistry

registry = AdapterRegistry()
registry.register_framework(MyCustomAdapter())
```
