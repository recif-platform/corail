"""ADK (Agent Development Kit) framework adapter — simplest prompt-response flow."""

from corail.adapters.llms.base import LLMAdapter
from corail.core.agent_config import AgentConfig
from corail.core.errors import LLMError


class ADKAdapter:
    """ADK adapter: system_prompt + user_input -> LLM -> response."""

    def supports(self, framework: str) -> bool:
        """Return True for 'adk' framework."""
        return framework == "adk"

    async def execute(self, config: AgentConfig, input_text: str, llm: LLMAdapter) -> str:
        """Execute agent by sending system_prompt + input to the LLM."""
        messages = [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": input_text},
        ]

        try:
            return await llm.generate(messages, config.model, config.temperature)
        except Exception as exc:
            raise LLMError(
                message=f"LLM call failed for agent {config.id}: {exc}",
                code="LLM_CALL_FAILED",
                details={"agent_id": config.id, "model": config.model, "provider": config.llm_provider},
            ) from exc
