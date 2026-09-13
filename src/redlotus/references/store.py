from __future__ import annotations

import asyncio
import hashlib
import mimetypes
import os
from pathlib import Path

from filelock import AsyncFileLock

from redlotus.infra.paths import user_data_dir
from redlotus.infra.persist_utils import atomic_write_json
from redlotus.ModelGateway.input_policy import ModelInputPolicy
from redlotus.references.models import ReferenceFile, ReferencePart
from redlotus.references.readers import DocumentReader
from redlotus.runtime.context import WorkspaceContext


class ReferenceStore:
    def __init__(self, workspace: WorkspaceContext, root: Path | None = None):
        self.workspace = workspace
        self.root = root or user_data_dir() / "references"

    async def prepare_message(self, message):
        import base64
        import mimetypes
        from pydantic_ai import BinaryContent, ImageUrl, VideoUrl
        from redlotus.ModelGateway.input_policy import ModelInputPolicy

        policy = ModelInputPolicy.for_role("coordinator")
        references = list(message.references)
        for index, item in enumerate(message.attachments):
            if isinstance(item, BinaryContent):
                name = item.identifier or f"attachment-{index}"
                if not Path(name).suffix:
                    name += mimetypes.guess_extension(item.media_type) or ".bin"
                ref = await self.import_bytes(
                    item.data, name=name, source=f"attachment:{name}", policy=policy
                )
            elif isinstance(item, (ImageUrl, VideoUrl)):
                if item.url.startswith("data:"):
                    header, encoded = item.url.split(",", 1)
                    mime = header[5:].split(";")[0]
                    name = "attachment" + (mimetypes.guess_extension(mime) or ".bin")
                    ref = await self.import_bytes(
                        base64.b64decode(encoded),
                        name=name,
                        source=f"attachment:{index}",
                        policy=policy,
                    )
                else:
                    ref = await self.import_url(item.url, policy=policy)
            else:
                raise ValueError(f"Unsupported attachment type: {type(item).__name__}")
            references.append(ref)
        references = list({ref.id: ref for ref in references}.values())
        policy.check([ref.byte_size for ref in references])
        message.references, message.attachments = references, []

    async def import_file(
        self, path: Path, *, policy: ModelInputPolicy
    ) -> ReferenceFile:
        policy.check([path.stat().st_size])
        data = await asyncio.to_thread(path.read_bytes)
        return await self.import_bytes(
            data, name=path.name, source=str(path), policy=policy
        )

    async def import_url(
        self, url: str, *, policy: ModelInputPolicy, media_type: str = ""
    ) -> ReferenceFile:
        import httpx
        from urllib.parse import urlsplit, unquote
        from redlotus.infra.shared_http import get_client

        if urlsplit(url).scheme not in ("https", "http"):
            raise ValueError("Remote references require HTTP(S)")
        client = get_client(
            "reference_download",
            lambda: httpx.AsyncClient(timeout=60, follow_redirects=True),
        )
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            data = bytearray()
            async for chunk in response.aiter_bytes():
                data.extend(chunk)
                policy.check([len(data)])
            mime = media_type or response.headers.get("content-type", "").split(";")[0]
        name = Path(unquote(urlsplit(url).path)).name or "attachment"
        if not Path(name).suffix:
            name += mimetypes.guess_extension(mime) or ".bin"
        return await self.import_bytes(
            bytes(data), name=name, source=url, policy=policy
        )

    async def import_bytes(
        self, data: bytes, *, name: str, source: str, policy: ModelInputPolicy
    ) -> ReferenceFile:
        policy.check([len(data)])
        digest = hashlib.sha256(data).hexdigest()
        identity = hashlib.sha256(
            f"{self.workspace.project_id}\0{os.path.normcase(source)}\0{digest}".encode()
        ).hexdigest()[:32]
        manifest = self.root / "manifests" / f"{identity}.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        async with AsyncFileLock(str(manifest) + ".lock", run_in_executor=False):
            if manifest.is_file():
                return ReferenceFile.model_validate_json(
                    manifest.read_text(encoding="utf-8")
                )
            directory = self.root / "blobs" / digest[:32]
            directory.mkdir(parents=True, exist_ok=True)
            snapshot = directory / ("source" + Path(name).suffix.lower())
            media_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
            async with AsyncFileLock(directory / ".build.lock", run_in_executor=False):
                if not snapshot.exists():
                    await asyncio.to_thread(snapshot.write_bytes, data)
                parts_path = directory / ("parts" + snapshot.suffix + ".json")
                if parts_path.is_file():
                    import json

                    parts = [
                        ReferencePart.model_validate(item)
                        for item in json.loads(parts_path.read_text(encoding="utf-8"))
                    ]
                elif media_type.startswith(("image/", "video/", "audio/")):
                    kind = media_type.split("/")[0]
                    native_path, native_type = snapshot, media_type
                    if kind == "image":
                        from PIL import Image

                        with Image.open(snapshot) as picture:
                            picture.verify()
                        if media_type not in (
                            "image/png",
                            "image/jpeg",
                            "image/webp",
                            "image/gif",
                        ):
                            native_path, native_type = (
                                directory / "image.png",
                                "image/png",
                            )
                            with Image.open(snapshot) as picture:
                                picture.save(native_path)
                    elif kind == "video":
                        header = data[:16]
                        if not (
                            header[4:8] == b"ftyp"
                            or header.startswith((b"RIFF", b"\x1aE\xdf\xa3"))
                        ):
                            raise ValueError(
                                "视频容器无效或未识别，不能作为原生视频提交。"
                            )
                    parts = [
                        ReferencePart(
                            kind=kind,
                            path=native_path,
                            media_type=native_type,
                            locator="原件",
                        )
                    ]
                    await asyncio.to_thread(
                        atomic_write_json,
                        parts_path,
                        [p.model_dump(mode="json") for p in parts],
                    )
                else:
                    parts = await DocumentReader().read(snapshot, directory)
                    await asyncio.to_thread(
                        atomic_write_json,
                        parts_path,
                        [p.model_dump(mode="json") for p in parts],
                    )
            reference = ReferenceFile(
                id=identity,
                project_id=self.workspace.project_id,
                name=name,
                source=source,
                media_type=media_type,
                byte_size=len(data),
                sha256=digest,
                snapshot=snapshot,
                parts=parts,
            )
            # The claim is already held on this path; write through a separate atomic operation.
            await asyncio.to_thread(atomic_write_json, manifest, reference.manifest())
            return reference

    def load(self, reference_id: str) -> ReferenceFile:
        if len(reference_id) != 32 or any(
            c not in "0123456789abcdef" for c in reference_id
        ):
            raise ValueError("引用文件 ID 无效。")
        path = self.root / "manifests" / f"{reference_id}.json"
        return ReferenceFile.model_validate_json(path.read_text(encoding="utf-8"))

    async def read_reference(self, reference_id: str):
        """Read a previously registered reference in this project, including native media."""
        from pydantic_ai import ToolReturn

        reference = await asyncio.to_thread(self.load, reference_id)
        if reference.project_id != self.workspace.project_id:
            return "Error: Reference belongs to another project; retrieve its authorized memory record instead."
        return ToolReturn(
            return_value=f"Read reference {reference.name} ({reference.id})",
            content=await asyncio.to_thread(reference.to_prompt),
        )
