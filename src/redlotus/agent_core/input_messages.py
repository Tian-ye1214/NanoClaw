from dataclasses import dataclass, field

from redlotus.references.models import ReferenceFile


@dataclass
class UserMessage:
    """User prompt text plus optional pydantic-AI multimodal content."""

    text: str
    attachments: list = field(default_factory=list)
    original_text: str | None = None
    references: list[ReferenceFile] = field(default_factory=list)

    def to_prompt(self):
        """Pass original requirements and explicitly labelled reference data together."""
        parts = [self.text]
        for reference in self.references:
            parts.extend(reference.to_prompt())
        parts.extend(self.attachments)
        return parts if len(parts) > 1 else self.text


def user_message_from_cli_input(raw_input: str) -> UserMessage:
    """Reference resolution happens once in the input controller, never in file contents."""
    return UserMessage(text=raw_input, original_text=raw_input)
