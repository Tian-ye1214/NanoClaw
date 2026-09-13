"""Gateway-owned input limits; model selection and sampling settings are untouched."""

from dataclasses import dataclass

from redlotus.config.app_config import settings


@dataclass(frozen=True)
class ModelInputPolicy:
    max_files: int = 20
    max_file_bytes: int = 20_000_000
    max_request_bytes: int | None = None

    @classmethod
    def for_role(cls, role: str = "coordinator") -> "ModelInputPolicy":
        limits = settings().get("input_limits", {})
        values = {**limits.get("defaults", {}), **limits.get(role, {})}
        return cls(
            max_file_bytes=int(values.get("max_file_bytes", 20_000_000)),
            max_request_bytes=values.get("max_request_bytes"),
        )

    def check(self, sizes: list[int]) -> None:
        if len(sizes) > self.max_files:
            raise ValueError(
                f"最多引用 {self.max_files} 个文件，本次为 {len(sizes)} 个。"
            )
        if any(size > self.max_file_bytes for size in sizes):
            raise ValueError(f"单个引用文件超过网关限额 {self.max_file_bytes:,} 字节。")
        if self.max_request_bytes is not None and sum(sizes) > self.max_request_bytes:
            raise ValueError(
                f"引用文件总量超过网关请求限额 {self.max_request_bytes:,} 字节。"
            )
