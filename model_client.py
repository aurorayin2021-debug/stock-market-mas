"""OpenAI 兼容 Chat Completions 公共调用。"""
from __future__ import annotations
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def text_content(value) -> str:
    """保留文本；其他可 JSON 序列化内容转为文本。"""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, allow_nan=False)


class ModelClient:
    def __init__(self, *, base_url: str, api_key: str, timeout: float = 60):
        if not base_url or not api_key:
            raise ValueError("需要配置 MODEL_BASE_URL 和 MODEL_API_KEY。")
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.timeout = timeout

    @classmethod
    def from_environment(cls):
        return cls(base_url=os.getenv("MODEL_BASE_URL", ""), api_key=os.getenv("MODEL_API_KEY", ""))

    def chat(self, *, model: str, messages: list, tools=None, tool_choice=None) -> dict:
        """执行一轮对话，返回 assistant 消息；工具由调用方实际执行。"""
        if not model or not messages:
            raise ValueError("模型名和消息均需填写。")
        payload = {"model": model, "messages": messages}
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        request = Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
            headers={"Authorization": "Bearer " + self._api_key, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            raise RuntimeError(f"模型 API 返回 HTTP {error.code}。") from error
        except URLError as error:
            raise RuntimeError("连接模型 API 失败，请检查网络和地址。") from error
        try:
            message = result["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("模型响应不包含预期的消息。") from error
        if not isinstance(message, dict):
            raise ValueError("模型响应的消息格式不正确。")
        return message

    def complete(self, *, model: str, prompt: str, content) -> str:
        if not model or not prompt or not prompt.strip():
            raise ValueError("模型名和 Prompt 均需填写后才能调用。")
        if not isinstance(content, (str, list)):
            raise TypeError("模型输入需为文本，或服务支持的多模态内容列表。")
        message = self.chat(model=model, messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": content},
        ])
        result = message.get("content")
        if not isinstance(result, str):
            raise ValueError("模型响应的内容不是文本，请检查接口格式。")
        return result
