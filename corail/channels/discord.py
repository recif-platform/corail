"""Discord channel — slash-command bot with streaming via message edits."""

import asyncio
import logging
import os
import re

import discord
from discord import app_commands

from corail.channels.base import Channel, get_collected_events, reset_events
from corail.config import Settings
from corail.core.pipeline import Pipeline
from corail.core.stream import StreamEvent
from corail.storage.factory import StorageFactory
from corail.storage.port import StoragePort

logger = logging.getLogger(__name__)

# Discord caps messages at 2000 chars. Leave room for "..." truncation.
_MAX_MSG_LEN = 1950

# Minimum interval between message edits to respect Discord rate limits.
_EDIT_INTERVAL = 1.5

# Strip <think>...</think> blocks — they render as noise in Discord.
_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.DOTALL)
_TOOL_TAG_RE = re.compile(r"</?tool_(?:use|result)>[\s\S]*?(?:</tool_(?:use|result)>|$)")
# Discord doesn't render markdown tables — convert to simple lines.
_MD_TABLE_RE = re.compile(r"^\|.*\|$", re.MULTILINE)
_MD_TABLE_SEP_RE = re.compile(r"^\|[-| :]+\|$", re.MULTILINE)


def _clean_for_discord(text: str) -> str:
    """Remove artefacts that only make sense in a web UI and convert
    markdown tables to plain text (Discord doesn't render them)."""
    text = _THINK_RE.sub("", text)
    text = _TOOL_TAG_RE.sub("", text)
    # Strip table separator rows (|---|---|)
    text = _MD_TABLE_SEP_RE.sub("", text)

    # Convert table rows: | A | B | C | → A  •  B  •  C
    def _table_row(m: re.Match) -> str:
        cells = [c.strip() for c in m.group(0).strip("|").split("|")]
        return "  •  ".join(c for c in cells if c)

    text = _MD_TABLE_RE.sub(_table_row, text)
    # Collapse multiple blank lines left by table removal.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class DiscordChannel(Channel):
    """Discord bot that exposes the Corail pipeline as slash commands.

    The agent responds to ``/chat <message>`` and streams the answer by
    editing the same Discord message every ~1.5 s. Conversations are
    persisted per user via the configured StoragePort so the agent remembers
    context across messages.
    """

    def __init__(self, pipeline: Pipeline, settings: Settings) -> None:
        super().__init__(pipeline, settings)

        self._storage: StoragePort | None = None
        self._trace_map: dict[int, str] = {}  # response message_id → MLflow trace_id

        intents = discord.Intents.default()
        intents.message_content = True
        intents.reactions = True

        self.bot = discord.Client(intents=intents)
        self.tree = app_commands.CommandTree(self.bot)

        # --- Events ---

        @self.bot.event
        async def on_raw_reaction_add(payload: discord.RawReactionActionEvent) -> None:
            if self.bot.user and payload.user_id == self.bot.user.id:
                return  # ignore the bot's own reactions
            trace_id = self._trace_map.get(payload.message_id)
            if not trace_id:
                return
            emoji = str(payload.emoji)
            if emoji == "👍":
                self.log_feedback(trace_id, True)
            elif emoji == "👎":
                self.log_feedback(trace_id, False)

        @self.bot.event
        async def on_ready() -> None:
            guild_id = os.environ.get("DISCORD_GUILD_ID")
            if guild_id:
                guild = discord.Object(id=int(guild_id))
                self.tree.copy_global_to(guild=guild)
                await self.tree.sync(guild=guild)
                logger.info("slash commands synced to guild %s", guild_id)
            else:
                await self.tree.sync()
                logger.info("slash commands synced globally (may take up to 1 h)")
            logger.info(
                "Discord bot ready — %s (%s)",
                self.bot.user,
                self.bot.user.id if self.bot.user else "?",
            )

        # --- Slash commands ---

        @self.tree.command(name="chat", description="Chat with the agent")
        @app_commands.describe(message="Your message to the agent")
        async def chat_cmd(interaction: discord.Interaction, message: str) -> None:
            await self._handle_chat(interaction, message)

        @self.tree.command(name="clear", description="Clear your conversation history")
        async def clear_cmd(interaction: discord.Interaction) -> None:
            await self._handle_clear(interaction)

        @self.tree.command(name="status", description="Show agent status")
        async def status_cmd(interaction: discord.Interaction) -> None:
            await self._handle_status(interaction)

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    @property
    def storage(self) -> StoragePort:
        if self._storage is None:
            self._storage = StorageFactory.create(self.settings.storage)
        return self._storage

    def _cid(self, user: discord.User | discord.Member) -> str:
        """Deterministic conversation id per Discord user."""
        return f"discord_{user.id}"

    # ------------------------------------------------------------------
    # /chat
    # ------------------------------------------------------------------

    async def _handle_chat(
        self,
        interaction: discord.Interaction,
        message: str,
    ) -> None:
        cid = self._cid(interaction.user)

        if not await self.storage.conversation_exists(cid):
            await self.storage.create_conversation(cid)

        history = await self.storage.get_messages(cid)
        await self.storage.append_message(cid, "user", message)

        # Acknowledge immediately — we'll follow up with the streamed answer.
        await interaction.response.defer(thinking=True)

        full_response = ""
        response_msg: discord.WebhookMessage | None = None
        last_edit = 0.0
        sources: list[dict] = []

        reset_events()
        try:
            async for token in self.pipeline.execute_stream(
                message,
                history=history,
            ):
                if isinstance(token, StreamEvent):
                    sse = token.to_sse_data()
                    if "sources" in sse:
                        sources.extend(sse["sources"])
                    continue

                if isinstance(token, str):
                    full_response += token

                    # Batch edits to stay under rate limits.
                    now = asyncio.get_event_loop().time()
                    if now - last_edit >= _EDIT_INTERVAL:
                        display = _clean_for_discord(full_response)[:_MAX_MSG_LEN]
                        if display:
                            if response_msg is None:
                                response_msg = await interaction.followup.send(
                                    display,
                                    wait=True,
                                )
                            else:
                                await response_msg.edit(content=display)
                            last_edit = now

            # Final edit with complete response.
            clean = _clean_for_discord(full_response)
            if not clean:
                clean = "(empty response)"

            # Only show sources if the LLM actually referenced them.
            # Avoids a misleading footer when the question is unrelated to
            # the KB and the retriever just returns noise.
            if sources and clean:
                resp_lower = clean.lower()
                seen: dict[str, int] = {}
                for src in sources:
                    seen[src["filename"]] = seen.get(src["filename"], 0) + 1
                used = {fn: n for fn, n in seen.items() if fn.rsplit(".", 1)[0].lower() in resp_lower}
                if used:
                    footer_parts = ["\n\n-# Sources:"]
                    for fn, n in used.items():
                        suffix = f" ({n} passages)" if n > 1 else ""
                        footer_parts.append(f" `{fn}`{suffix}")
                    footer = "".join(footer_parts)
            else:
                footer = ""

            final_text = (clean + footer)[:2000]

            if response_msg is None:
                await interaction.followup.send(final_text)
            else:
                await response_msg.edit(content=final_text)

            # Persist.
            await self.storage.append_message(cid, "assistant", clean)

            # Trace to MLflow and attach 👍/👎 reactions so users can rate the response.
            trace_id = await self.log_chat_trace(message, cid, clean, get_collected_events())
            logger.info("MLflow trace_id for message: %s", trace_id)
            if trace_id and response_msg is not None:
                self._trace_map[response_msg.id] = trace_id
                try:
                    await response_msg.add_reaction("👍")
                    await response_msg.add_reaction("👎")
                    logger.info("Reactions added to message %s", response_msg.id)
                except Exception as e:
                    logger.warning("Failed to add reactions: %s", e)

        except Exception as exc:
            logger.exception("Discord chat error for user %s", interaction.user)
            error_text = f"Something went wrong: {type(exc).__name__}"
            if response_msg is None:
                await interaction.followup.send(error_text)
            else:
                await response_msg.edit(content=error_text)

    # ------------------------------------------------------------------
    # /clear
    # ------------------------------------------------------------------

    async def _handle_clear(self, interaction: discord.Interaction) -> None:
        cid = self._cid(interaction.user)
        if await self.storage.conversation_exists(cid):
            await self.storage.delete_conversation(cid)
        await interaction.response.send_message(
            "Conversation cleared.",
            ephemeral=True,
        )

    # ------------------------------------------------------------------
    # /status
    # ------------------------------------------------------------------

    async def _handle_status(self, interaction: discord.Interaction) -> None:
        agent_name = self.settings.agent_name or "agent"
        model = f"{self.settings.model_type}/{self.settings.model_id}"
        embed = discord.Embed(
            title=agent_name,
            color=discord.Color.teal(),
        )
        embed.add_field(name="Model", value=f"`{model}`", inline=True)
        embed.add_field(name="Strategy", value=f"`{self.settings.strategy}`", inline=True)
        embed.add_field(name="Storage", value=f"`{self.settings.storage}`", inline=True)
        await interaction.response.send_message(embed=embed)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        token = os.environ.get("DISCORD_BOT_TOKEN")
        if not token:
            raise ValueError(
                "DISCORD_BOT_TOKEN env var not set. Create a bot at "
                "https://discord.com/developers/applications and set the token."
            )
        logger.info("Starting Discord bot…")
        self.bot.run(token, log_handler=None)

    def stop(self) -> None:
        if self.bot.loop and self.bot.loop.is_running():
            asyncio.ensure_future(self.bot.close())
