"""Native video content for OpenAI-compatible multimodal gateways."""

from pydantic_ai.messages import BinaryContent, VideoUrl
from pydantic_ai.models.openai import OpenAIChatModel


class MultimodalChatModel(OpenAIChatModel):
    def _map_model_response(self, message):
        mapped = super()._map_model_response(message)
        if mapped and not mapped.get("content") and not mapped.get("tool_calls"):
            return None  # Interrupted thinking has no completed assistant message to replay.
        if mapped and mapped.get("tool_calls") and mapped.get("content") is None:
            mapped["content"] = ""
        return mapped

    async def _map_video_url_item(self, item: VideoUrl):
        return {"type": "video_url", "video_url": {"url": item.url}}

    async def _map_binary_content_item(self, item: BinaryContent):
        if item.is_video:
            if "deepseek" in self.model_name.lower():
                return {
                    "type": "file",
                    "filename": "reference.mp4",
                    "file_data": item.data_uri,
                }
            return {"type": "video_url", "video_url": {"url": item.data_uri}}
        return await super()._map_binary_content_item(item)
