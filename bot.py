# -*- coding: utf-8 -*-
"""Надёжный Telegram-бот для работы с OpenRouter.

Основные улучшения по сравнению с исходной версией:
* Чёткая конфигурация и валидация переменных окружения.
* Асинхронные вызовы OpenRouter с повторными попытками и тайм-аутами.
* Безопасное хранение пользовательского состояния с блокировками.
* Единый код тримминга контекста и оценка токенов.
* Разделение логики на отдельные классы (конфигурация, состояние, клиент).
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import math
import os
import time
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from io import BytesIO
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from html import unescape

import httpx
from dotenv import load_dotenv
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
    Update,
)
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ---------------------------------------------------------------------------
# Конфигурация и логирование
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

LOG = logging.getLogger(__name__)


class ConfigError(RuntimeError):
    """Исключение при проблемах с конфигурацией."""


@dataclass(frozen=True)
class Config:
    """Все параметры запуска бота."""

    bot_token: str
    api_key: str
    base_url: str
    available_models: Tuple[str, ...]
    default_model: Optional[str]
    http_referer: Optional[str]
    title: Optional[str]
    context_max_tokens: int = 8_000
    reply_max_chars: int = 4_000
    min_interval_sec: float = 3.0
    openrouter_timeout: float = 45.0
    openrouter_retries: int = 2

    @classmethod
    def from_env(cls) -> "Config":
        bot_token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
        api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
        base_url = (os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").strip().rstrip("/")
        models_csv = (os.getenv("OPENROUTER_MODELS") or "").strip()
        default_model = (os.getenv("OPENROUTER_DEFAULT_MODEL") or "").strip() or None
        http_referer = (os.getenv("OPENROUTER_HTTP_REFERER") or "").strip() or None
        title = (os.getenv("OPENROUTER_TITLE") or "").strip() or None

        if not bot_token:
            raise ConfigError("В .env нет TELEGRAM_BOT_TOKEN")
        if not api_key:
            raise ConfigError("В .env отсутствует OPENROUTER_API_KEY")

        models: Tuple[str, ...] = tuple(m.strip() for m in models_csv.split(",") if m.strip())

        return cls(
            bot_token=bot_token,
            api_key=api_key,
            base_url=base_url,
            available_models=models,
            default_model=default_model,
            http_referer=http_referer,
            title=title,
        )


# ---------------------------------------------------------------------------
# Оценка токенов и работа с контекстом
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = "Ты дружелюбный помощник. Отвечай кратко по-русски."


def estimate_tokens(text: str) -> int:
    """Грубая оценка количества токенов."""

    if not text:
        return 0
    return max(1, int(math.ceil(len(text) / 3.7)))


def estimate_messages_tokens(messages: Sequence[Dict[str, str]]) -> int:
    total = 0
    for message in messages:
        total += estimate_tokens(message.get("content", ""))
        total += 4  # накладные расходы на роль/формат
    return total


def clip_text(text: str, limit: int) -> str:
    """Обрезает текст до лимита символов."""

    if len(text) <= limit:
        return text
    trimmed = text[:limit].rstrip()
    return f"{trimmed}\n…обрезано…"


def _format_code_block(match: re.Match) -> str:
    """Форматирует многострочный код-блок без Markdown-разметки."""

    code = (match.group(2) or "").strip("\n")
    if not code:
        return ""
    lines = code.splitlines()
    formatted = "\n".join(f"    {line}" if line else "" for line in lines)
    return f"\n{formatted}\n"


def format_for_display(text: str) -> str:
    """Приводит ответ модели к удобному для чтения виду без лишних символов."""

    if not text:
        return ""

    normalized = unescape(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return ""

    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = re.sub(r"```(\w+)?\n([\s\S]*?)```", _format_code_block, normalized)

    normalized = re.sub(r"`([^`]+)`", r"«\1»", normalized)
    normalized = re.sub(r"\*\*([^*]+)\*\*", r"\1", normalized)
    normalized = re.sub(r"__([^_]+)__", r"\1", normalized)
    normalized = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", normalized)
    normalized = re.sub(r"(?<!_)_([^_]+)_(?!_)", r"\1", normalized)
    normalized = re.sub(r"~~([^~]+)~~", r"\1", normalized)
    normalized = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 — \2", normalized)

    normalized = re.sub(r"(?m)^>\s?", "Цитата: ", normalized)
    normalized = re.sub(r"(?m)^#{1,6}\s*", "", normalized)
    normalized = re.sub(r"(?m)^[*-]\s+", "• ", normalized)

    normalized = re.sub(r"\xa0", " ", normalized)
    normalized = re.sub(r" {2,}", " ", normalized)
    normalized = re.sub(r"\n +", "\n", normalized)

    return normalized


def prepare_display_pair(text: str, limit: int) -> Tuple[str, str]:
    """Возвращает пару из текста для истории и текста для пользователя."""

    raw = clip_text(text, limit)
    pretty = format_for_display(raw)
    if not pretty:
        pretty = "(пустой ответ)"
    return raw, pretty


def trim_context(messages: List[Dict[str, str]], max_tokens: int) -> Tuple[List[Dict[str, str]], int]:
    """Ограничивает историю диалога по количеству токенов."""

    if not messages:
        return messages, 0

    total = estimate_messages_tokens(messages)
    if total <= max_tokens:
        return messages, total

    system_msg: Optional[Dict[str, str]] = None
    start = 0
    if messages[0].get("role") == "system":
        system_msg = messages[0]
        start = 1

    tail = messages[start:]
    keep_min = min(len(tail), 8)
    cut = 0

    while True:
        candidate = ([system_msg] if system_msg else []) + tail[cut:]
        total = estimate_messages_tokens(candidate)
        if total <= max_tokens or (len(tail) - cut) <= keep_min:
            return candidate, total
        cut += 1


def context_stats(messages: Sequence[Dict[str, str]], max_tokens: int) -> Tuple[int, float]:
    used = estimate_messages_tokens(messages)
    percentage = 0.0 if max_tokens <= 0 else min(100.0, (used / max_tokens) * 100.0)
    return used, percentage


# ---------------------------------------------------------------------------
# Состояние пользователей
# ---------------------------------------------------------------------------


@dataclass
class UserSession:
    """История пользователя."""

    messages: List[Dict[str, str]] = field(default_factory=lambda: [
        {"role": "system", "content": SYSTEM_PROMPT}
    ])
    model: Optional[str] = None
    last_use: float = 0.0

    def ensure_model(self, default_model: Optional[str]) -> Optional[str]:
        if self.model:
            return self.model
        self.model = default_model
        return self.model


class BotState:
    """Потокобезопасное хранилище пользовательского состояния."""

    def __init__(self) -> None:
        self._sessions: Dict[int, UserSession] = {}
        self._locks: Dict[int, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    def get_session(self, user_id: int) -> UserSession:
        session = self._sessions.get(user_id)
        if not session:
            session = UserSession()
            self._sessions[user_id] = session
        return session

    def reset_context(self, user_id: int) -> None:
        session = self.get_session(user_id)
        session.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def set_model(self, user_id: int, model: str) -> None:
        session = self.get_session(user_id)
        session.model = model

    @asynccontextmanager
    async def user_lock(self, user_id: int):
        async with self._global_lock:
            lock = self._locks.setdefault(user_id, asyncio.Lock())
        await lock.acquire()
        try:
            yield
        finally:
            lock.release()


# ---------------------------------------------------------------------------
# Работа с OpenRouter
# ---------------------------------------------------------------------------


@dataclass
class OpenRouterResult:
    type: str
    text: str = ""
    images: Tuple[str, ...] = ()

    @classmethod
    def text_result(cls, text: str) -> "OpenRouterResult":
        return cls(type="text", text=text)

    @classmethod
    def images_result(cls, images: Iterable[str], text: str) -> "OpenRouterResult":
        return cls(type="images", images=tuple(images), text=text)

    @classmethod
    def error(cls, text: str) -> "OpenRouterResult":
        return cls(type="error", text=text)


class OpenRouterClient:
    """Асинхронный клиент OpenRouter."""

    def __init__(self, config: Config) -> None:
        self._config = config
        timeout = httpx.Timeout(config.openrouter_timeout)
        self._client = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        await self._client.aclose()

    async def generate(
        self,
        model: Optional[str],
        messages: Sequence[Dict[str, str]],
        need_image: bool,
    ) -> OpenRouterResult:
        if not model:
            return OpenRouterResult.error("Модель не выбрана. Нажмите «Выбрать модель».")

        safe_messages = [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in messages
        ]

        payload = {
            "model": model,
            "messages": safe_messages,
            "temperature": 0.7,
        }
        if need_image:
            payload["modalities"] = ["image", "text"]

        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }
        if self._config.http_referer:
            headers["HTTP-Referer"] = self._config.http_referer
        if self._config.title:
            headers["X-Title"] = self._config.title

        url = f"{self._config.base_url}/chat/completions"

        for attempt in range(1, self._config.openrouter_retries + 2):
            try:
                response = await self._client.post(url, headers=headers, json=payload)
            except httpx.RequestError as exc:
                if attempt > self._config.openrouter_retries:
                    return OpenRouterResult.error(f"OpenRouter сеть/таймаут: {exc}")
                await asyncio.sleep(1.0)
                continue

            if response.status_code == 401:
                return OpenRouterResult.error("OpenRouter 401: ключ не принят (проверьте OPENROUTER_API_KEY).")
            if response.status_code == 402:
                return OpenRouterResult.error(
                    "OpenRouter 402: бесплатная очередь/лимиты. Попробуйте позже или смените модель."
                )
            if response.status_code != 200:
                text = response.text[:300]
                if attempt > self._config.openrouter_retries:
                    return OpenRouterResult.error(f"OpenRouter HTTP {response.status_code}: {text}")
                await asyncio.sleep(1.0)
                continue

            try:
                data = response.json()
            except ValueError as exc:
                return OpenRouterResult.error(f"OpenRouter парсинг JSON: {exc}")

            message = (data.get("choices") or [{}])[0].get("message", {}) or {}
            images = tuple(
                url_data
                for item in message.get("images", [])
                for url_data in [((item.get("image_url") or {}).get("url"))]
                if isinstance(url_data, str) and url_data.startswith("data:image")
            )

            if images:
                caption = (message.get("content") or "").strip()
                return OpenRouterResult.images_result(images, caption)

            content = (message.get("content") or "").strip()
            if content:
                return OpenRouterResult.text_result(content)

            return OpenRouterResult.error("Не удалось извлечь ответ модели.")

        return OpenRouterResult.error("Не удалось получить ответ от OpenRouter.")


# ---------------------------------------------------------------------------
# Клавиатуры
# ---------------------------------------------------------------------------


HELP_TEXT = (
    "Как одобрить pull request на GitHub:\n"
    "1. Откройте вкладку Pull requests нужного репозитория и выберите нужный PR.\n"
    "2. Перейдите на вкладку Files changed и нажмите Review changes.\n"
    "3. В открывшемся окне выберите пункт Approve, при необходимости оставьте комментарий и нажмите Submit review.\n\n"
    "Если вы просто хотите подтвердить, что ответ бота подошёл, напишите «Одобряю» или отправьте 👍 — я отмечу, что всё прошло успешно."
)


def main_menu_keyboard() -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton("🤖 Выбрать модель")],
        [KeyboardButton("🗑️ Очистить контекст"), KeyboardButton("📊 Контекст: состояние")],
        [KeyboardButton("ℹ️ Помощь / как одобрить")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)


def models_inline_keyboard(models: Sequence[str]) -> InlineKeyboardMarkup:
    if not models:
        return InlineKeyboardMarkup(
            [[InlineKeyboardButton("Список моделей пуст — как добавить?", callback_data="models_empty")]]
        )

    rows: List[List[InlineKeyboardButton]] = []
    row: List[InlineKeyboardButton] = []
    for model in models:
        title = model if len(model) <= 32 else model[:29] + "…"
        row.append(InlineKeyboardButton(title, callback_data=f"set_model::{model}"))
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    return InlineKeyboardMarkup(rows)


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def wants_image(user_text: str, model_id: Optional[str]) -> bool:
    text = (user_text or "").lower()
    keywords = [
        "нарисуй",
        "картинку",
        "картинка",
        "рисунок",
        "изображение",
        "image",
        "illustration",
        "art",
    ]
    if any(keyword in text for keyword in keywords):
        return True
    model = (model_id or "").lower()
    return "image" in model or "flash-image" in model


def get_app_components(context: ContextTypes.DEFAULT_TYPE) -> Tuple[Config, BotState, OpenRouterClient]:
    data = context.application.bot_data
    return data["config"], data["state"], data["client"]


# ---------------------------------------------------------------------------
# Обработчики Telegram
# ---------------------------------------------------------------------------


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    config, state, _ = get_app_components(context)
    user_id = update.effective_user.id

    session = state.get_session(user_id)
    session.ensure_model(config.default_model)

    await update.message.reply_text(
        "Привет! Я бот с ИИ через OpenRouter.\n"
        "1) Нажмите «🤖 Выбрать модель» и выберите одну из доступных.\n"
        "2) Пишите сообщения — отвечу выбранной моделью.\n"
        "Кнопки: «🗑️ Очистить контекст», «📊 Контекст: состояние», «ℹ️ Помощь / как одобрить».",
        reply_markup=main_menu_keyboard(),
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        HELP_TEXT,
        disable_web_page_preview=True,
        reply_markup=main_menu_keyboard(),
    )


async def show_models(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    config, _, _ = get_app_components(context)
    text = "Выберите модель из списка ниже."
    if not config.available_models:
        text = (
            "Список моделей пуст.\n\n"
            "Добавьте в .env строку OPENROUTER_MODELS в формате CSV, например:\n"
            "OPENROUTER_MODELS=provider/modelA,provider/modelB:free\n"
            "Затем перезапустите бота."
        )
    await update.message.reply_text(text, reply_markup=models_inline_keyboard(config.available_models))


async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    config, state, _ = get_app_components(context)
    query = update.callback_query
    if not query:
        return

    data = query.data or ""
    user_id = query.from_user.id

    if data == "models_empty":
        await query.answer()
        await query.edit_message_text(
            "OPENROUTER_MODELS пуст. Заполните .env и перезапустите бота."
        )
        return

    if data.startswith("set_model::"):
        model = data.split("set_model::", 1)[1]
        state.set_model(user_id, model)
        await query.answer("Модель выбрана ✅", show_alert=False)
        await query.edit_message_text(
            f"Вы выбрали модель:\n{model}\nТеперь отправьте сообщение — я отвечу."
        )
        return

    await query.answer()


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    config, state, client = get_app_components(context)
    user_id = update.effective_user.id
    text = update.message.text.strip()
    normalized = text.lower()

    async with state.user_lock(user_id):
        session = state.get_session(user_id)
        session.ensure_model(config.default_model)

        now = time.monotonic()
        if now - session.last_use < config.min_interval_sec:
            await update.message.reply_text(
                "Слишком часто. Подождите пару секунд 🙏",
                reply_markup=main_menu_keyboard(),
            )
            return
        session.last_use = now

        if text.startswith("🤖"):
            await show_models(update, context)
            return
        if text.startswith("🗑️"):
            state.reset_context(user_id)
            await update.message.reply_text(
                "Контекст очищен ✅", reply_markup=main_menu_keyboard()
            )
            return
        if text.startswith("📊"):
            used, pct = context_stats(session.messages, config.context_max_tokens)
            await update.message.reply_text(
                f"Контекст: ~{used} токенов из {config.context_max_tokens} (~{pct:.1f}%)",
                reply_markup=main_menu_keyboard(),
            )
            return
        if text.startswith("ℹ️") or "как одобрить" in normalized:
            await update.message.reply_text(
                HELP_TEXT,
                disable_web_page_preview=True,
                reply_markup=main_menu_keyboard(),
            )
            return
        if normalized in {"одобряю", "одобрено", "approved", "👍"} or normalized.startswith("одобряю"):
            await update.message.reply_text(
                "Спасибо за одобрение! Продолжаем работать 🤝",
                reply_markup=main_menu_keyboard(),
            )
            return

        model = session.model or ""
        if not model:
            await update.message.reply_text(
                "Сначала выберите модель: нажмите «🤖 Выбрать модель».",
                reply_markup=main_menu_keyboard(),
            )
            return

        session.messages.append({"role": "user", "content": text})
        session.messages, _ = trim_context(session.messages, config.context_max_tokens)

        placeholder = await update.message.reply_text("Думаю…")
        need_image = wants_image(text, model)

        try:
            result = await client.generate(model, session.messages, need_image)
        except Exception as exc:  # непредвиденные ошибки
            LOG.exception("Ошибка при обращении к OpenRouter")
            result = OpenRouterResult.error(f"Внутренняя ошибка: {exc}")

        if result.type == "images":
            try:
                await placeholder.delete()
            except Exception:
                pass

            caption = result.text or "Готово ✅"
            raw_caption, display_caption = prepare_display_pair(caption, config.reply_max_chars)
            for data_url in result.images:
                try:
                    payload = data_url.split(",", 1)[1] if "," in data_url else data_url
                    raw = base64.b64decode(payload)
                except (IndexError, ValueError, binascii.Error):
                    LOG.warning("Некорректный image data_url от OpenRouter")
                    continue

                bio = BytesIO(raw)
                bio.name = "image.png"
                await update.message.reply_photo(photo=bio)

            session.messages.append({"role": "assistant", "content": raw_caption})
            await update.message.reply_text(
                display_caption,
                reply_markup=main_menu_keyboard(),
            )
            return

        if result.type == "text":
            raw_text, display_text = prepare_display_pair(result.text, config.reply_max_chars)
            session.messages.append({"role": "assistant", "content": raw_text})
            try:
                await context.bot.edit_message_text(
                    chat_id=placeholder.chat_id,
                    message_id=placeholder.message_id,
                    text=display_text,
                    disable_web_page_preview=True,
                )
            except Exception:
                await update.message.reply_text(
                    display_text,
                    disable_web_page_preview=True,
                    reply_markup=main_menu_keyboard(),
                )
            return

        raw_error, display_error = prepare_display_pair(result.text or "Ошибка", config.reply_max_chars)
        session.messages.append({"role": "assistant", "content": raw_error})
        try:
            await context.bot.edit_message_text(
                chat_id=placeholder.chat_id,
                message_id=placeholder.message_id,
                text=display_error,
                disable_web_page_preview=True,
            )
        except Exception:
            await update.message.reply_text(
                display_error,
                disable_web_page_preview=True,
                reply_markup=main_menu_keyboard(),
            )


# ---------------------------------------------------------------------------
# Инициализация приложения
# ---------------------------------------------------------------------------


def main() -> None:
    config = Config.from_env()

    async def _post_init(application: Application) -> None:
        application.bot_data["config"] = config
        application.bot_data["state"] = BotState()
        application.bot_data["client"] = OpenRouterClient(config)
        LOG.info("Бот инициализирован")

    async def _post_shutdown(application: Application) -> None:
        client: OpenRouterClient = application.bot_data.get("client")
        if client:
            await client.close()
        LOG.info("Бот остановлен")

    application = (
        ApplicationBuilder()
        .token(config.bot_token)
        .post_init(_post_init)
        .post_shutdown(_post_shutdown)
        .build()
    )

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("help", cmd_help))
    application.add_handler(CallbackQueryHandler(on_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    LOG.info("Бот запущен (long polling). Ctrl+C — остановка.")
    application.run_polling()


if __name__ == "__main__":
    main()
