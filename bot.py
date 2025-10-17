# -*- coding: utf-8 -*-
"""
Telegram-бот (long polling, Windows 11 OK) для работы с OpenRouter.
Требования:
- Меню с тремя кнопками: "Выбрать модель", "Очистить контекст", "Контекст: состояние".
- Выбор модели только из OPENROUTER_MODELS (CSV). После выбора — подтверждение.
- Ответы выбранного ИИ на любой текст.
- Память диалога и тримминг по лимиту токенов.
- Оценка заполнения контекста.
- Поддержка генерации изображений (если модель умеет ИЛИ пользователь явно просит картинку).
- Никакого форматирования (parse_mode не используем) — только чистый текст.
"""

from __future__ import annotations
import os, time, math, asyncio, logging, base64
from io import BytesIO
from typing import Dict, List, Tuple

import requests
from dotenv import load_dotenv
from telegram import (
    Update,
    ReplyKeyboardMarkup,
    KeyboardButton,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

# -------------------- ENV & логирование --------------------
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("__main__")

BOT_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
OR_API_KEY = (os.getenv("OPENROUTER_API_KEY") or "").strip()
OR_BASE_URL = (os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").rstrip("/")
OR_MODELS_CSV = (os.getenv("OPENROUTER_MODELS") or "").strip()
OR_DEFAULT_MODEL = (os.getenv("OPENROUTER_DEFAULT_MODEL") or "").strip()
OR_HTTP_REFERER = (os.getenv("OPENROUTER_HTTP_REFERER") or "").strip()
OR_TITLE = (os.getenv("OPENROUTER_TITLE") or "").strip()

if not BOT_TOKEN:
    raise RuntimeError("В .env нет TELEGRAM_BOT_TOKEN")
if not OR_API_KEY:
    raise RuntimeError("В .env отсутствует OPENROUTER_API_KEY")

AVAILABLE_MODELS: List[str] = [m.strip() for m in OR_MODELS_CSV.split(",") if m.strip()]

# -------------------- состояние --------------------
UserId = int
USER_MODEL: Dict[UserId, str] = {}
USER_CONTEXT: Dict[UserId, List[Dict[str, str]]] = {}
LAST_USE: Dict[UserId, float] = {}
MIN_INTERVAL_SEC = 3.0

# лимиты контекста (эвристика)
CONTEXT_MAX_TOKENS = 8000
REPLY_MAX_CHARS = 4000

# -------------------- утилиты --------------------
def clip(text: str, n: int = REPLY_MAX_CHARS) -> str:
    return text if len(text) <= n else text[:n] + "\n\n…обрезано…"

def estimate_tokens(text: str) -> int:
    if not text: return 0
    return max(1, int(math.ceil(len(text) / 3.7)))  # грубая оценка

def estimate_messages_tokens(messages: List[Dict[str, str]]) -> int:
    total = 0
    for m in messages:
        total += estimate_tokens(m.get("content", ""))
        total += 4  # на разметку/роль
    return total

def trim_context(messages: List[Dict[str, str]], max_tokens: int) -> Tuple[List[Dict[str, str]], int]:
    if not messages:
        return messages, 0
    total = estimate_messages_tokens(messages)
    if total <= max_tokens:
        return messages, total

    system_msg = None
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

def context_stats(messages: List[Dict[str, str]], max_tokens: int) -> Tuple[int, float]:
    used = estimate_messages_tokens(messages)
    pct = min(100.0, (used / max_tokens) * 100.0 if max_tokens > 0 else 0.0)
    return used, pct

def wants_image(user_text: str, model_id: str) -> bool:
    t = (user_text or "").lower()
    if any(k in t for k in ["нарисуй", "картинку", "картинка", "рисунок", "изображение", "image", "illustration", "art"]):
        return True
    m = (model_id or "").lower()
    return ("image" in m) or ("flash-image" in m)

def main_menu_keyboard() -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton("🤖 Выбрать модель")],
        [KeyboardButton("🗑️ Очистить контекст"), KeyboardButton("📊 Контекст: состояние")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

def models_inline_keyboard(models: List[str]) -> InlineKeyboardMarkup:
    if not models:
        return InlineKeyboardMarkup([[InlineKeyboardButton("Список моделей пуст — как добавить?", callback_data="models_empty")]])
    rows, row = [], []
    for m in models:
        title = m if len(m) <= 32 else m[:29] + "…"
        row.append(InlineKeyboardButton(title, callback_data=f"set_model::{m}"))
        if len(row) == 2:
            rows.append(row); row = []
    if row: rows.append(row)
    return InlineKeyboardMarkup(rows)

# -------------------- OpenRouter --------------------
def call_openrouter(model: str, messages: List[Dict[str, str]], need_image: bool, timeout_sec: int = 45):
    """
    Возвращает dict:
      {"type": "text", "text": "..."} ИЛИ
      {"type": "images", "images": [data_url,...], "text": caption} ИЛИ
      {"type": "error", "text": "..."}
    """
    if not model:
        return {"type": "error", "text": "Модель не выбрана. Нажмите «Выбрать модель»."}

    url = f"{OR_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {OR_API_KEY}", "Content-Type": "application/json"}
    if OR_HTTP_REFERER: headers["HTTP-Referer"] = OR_HTTP_REFERER
    if OR_TITLE: headers["X-Title"] = OR_TITLE

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
    }
    if need_image:
        payload["modalities"] = ["image", "text"]

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)
    except requests.exceptions.RequestException as e:
        return {"type": "error", "text": f"OpenRouter сеть/таймаут: {e}"}

    if resp.status_code == 401:
        return {"type": "error", "text": "OpenRouter 401: ключ не принят (проверьте OPENROUTER_API_KEY)."}
    if resp.status_code == 402:
        return {"type": "error", "text": "OpenRouter 402: бесплатная очередь/лимиты. Попробуйте позже или смените модель."}
    if resp.status_code != 200:
        return {"type": "error", "text": f"OpenRouter HTTP {resp.status_code}: {resp.text[:300]}"}

    try:
        data = resp.json()
    except Exception as e:
        return {"type": "error", "text": f"OpenRouter парсинг JSON: {e}"}

    msg = (data.get("choices") or [{}])[0].get("message", {}) or {}

    # Если пришли изображения
    images = msg.get("images") or []
    urls = []
    for im in images:
        url_data = (im.get("image_url") or {}).get("url")
        if isinstance(url_data, str) and url_data.startswith("data:image"):
            urls.append(url_data)
    if urls:
        caption = (msg.get("content") or "").strip()
        return {"type": "images", "images": urls, "text": caption}

    # Иначе — текст
    content = (msg.get("content") or "").strip()
    if content:
        return {"type": "text", "text": content}

    return {"type": "error", "text": "Не удалось извлечь ответ модели."}

# -------------------- хендлеры --------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    # Инициализируем контекст и дефолтную модель (если задана)
    if uid not in USER_CONTEXT:
        USER_CONTEXT[uid] = [{"role": "system", "content": "Ты дружелюбный помощник. Отвечай кратко по-русски."}]
    if uid not in USER_MODEL and OR_DEFAULT_MODEL:
        USER_MODEL[uid] = OR_DEFAULT_MODEL

    await update.message.reply_text(
        "Привет! Я бот с ИИ через OpenRouter.\n"
        "1) Нажмите «🤖 Выбрать модель» и выберите одну из доступных.\n"
        "2) Пишите сообщения — отвечу выбранной моделью.\n"
        "Кнопки: «🗑️ Очистить контекст», «📊 Контекст: состояние».",
        reply_markup=main_menu_keyboard()
    )

async def show_models(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = "Выберите модель из списка ниже."
    if not AVAILABLE_MODELS:
        text = (
            "Список моделей пуст.\n\n"
            "Добавьте в .env строку OPENROUTER_MODELS в формате CSV, например:\n"
            "OPENROUTER_MODELS=provider/modelA,provider/modelB:free\n"
            "Затем перезапустите бота."
        )
    await update.message.reply_text(text, reply_markup=models_inline_keyboard(AVAILABLE_MODELS))

async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    if not q: return
    data = q.data or ""
    uid = q.from_user.id

    if data == "models_empty":
        await q.answer()
        await q.edit_message_text(
            "OPENROUTER_MODELS пуст. Заполните .env и перезапустите бота."
        )
        return

    if data.startswith("set_model::"):
        model = data.split("set_model::", 1)[1]
        USER_MODEL[uid] = model
        await q.answer("Модель выбрана ✅", show_alert=False)
        await q.edit_message_text(f"Вы выбрали модель:\n{model}\nТеперь отправьте сообщение — я отвечу.")
        return

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    uid = update.effective_user.id
    text = update.message.text.strip()

    # Анти-спам
    now = time.time()
    last = LAST_USE.get(uid, 0.0)
    if now - last < MIN_INTERVAL_SEC:
        await update.message.reply_text("Слишком часто. Подождите пару секунд 🙏", reply_markup=main_menu_keyboard())
        return
    LAST_USE[uid] = now

    # Обработка кнопок меню (ReplyKeyboard)
    if text.startswith("🤖"):
        await show_models(update, context)
        return
    if text.startswith("🗑️"):
        USER_CONTEXT[uid] = [{"role": "system", "content": "Ты дружелюбный помощник. Отвечай кратко по-русски."}]
        await update.message.reply_text("Контекст очищен ✅", reply_markup=main_menu_keyboard())
        return
    if text.startswith("📊"):
        msgs = USER_CONTEXT.get(uid, [])
        used, pct = context_stats(msgs, CONTEXT_MAX_TOKENS)
        await update.message.reply_text(f"Контекст: ~{used} токенов из {CONTEXT_MAX_TOKENS} (~{pct:.1f}%)", reply_markup=main_menu_keyboard())
        return

    # Проверим выбранную модель
    model = USER_MODEL.get(uid, "").strip()
    if not model:
        await update.message.reply_text("Сначала выберите модель: нажмите «🤖 Выбрать модель».", reply_markup=main_menu_keyboard())
        return

    # Инициализация контекста при необходимости
    USER_CONTEXT.setdefault(uid, [{"role": "system", "content": "Ты дружелюбный помощник. Отвечай кратко по-русски."}])

    # Добавляем запрос пользователя и триммим историю
    USER_CONTEXT[uid].append({"role": "user", "content": text})
    USER_CONTEXT[uid], _ = trim_context(USER_CONTEXT[uid], CONTEXT_MAX_TOKENS)

    # Индикация генерации
    placeholder = await update.message.reply_text("Думаю…")

    need_image = wants_image(text, model)

    # Запрос к OpenRouter в пуле
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, call_openrouter, model, USER_CONTEXT[uid], need_image)

    # Мягкий повтор при временных сетевых сбоях
    if result.get("type") == "error" and any(k in result.get("text","").lower() for k in ["таймаут", "temporari", "timeout", "econnreset", "enotfound"]):
        await asyncio.sleep(1.2)
        result = await loop.run_in_executor(None, call_openrouter, model, USER_CONTEXT[uid], need_image)

    # Обработка результата
    if result.get("type") == "images":
        # Удалим плейсхолдер
        try:
            await placeholder.delete()
        except Exception:
            pass

        caption = result.get("text") or "Готово ✅"
        # Отправляем каждую картинку
        for data_url in result.get("images", []):
            b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
            raw = base64.b64decode(b64)
            bio = BytesIO(raw); bio.name = "image.png"
            await update.message.reply_photo(photo=bio)

        USER_CONTEXT[uid].append({"role": "assistant", "content": caption})
        await update.message.reply_text(clip(caption), reply_markup=main_menu_keyboard())
        return

    if result.get("type") == "text":
        text_out = clip(result.get("text",""))
        USER_CONTEXT[uid].append({"role": "assistant", "content": text_out})
        # редактируем плейсхолдер готовым ответом (без parse_mode)
        try:
            await context.bot.edit_message_text(
                chat_id=placeholder.chat_id,
                message_id=placeholder.message_id,
                text=text_out,
                disable_web_page_preview=True,
            )
        except Exception:
            # если не получилось — отправим отдельным сообщением
            await update.message.reply_text(text_out, disable_web_page_preview=True, reply_markup=main_menu_keyboard())
        return

    # Ошибка
    err = clip(result.get("text","Ошибка"))
    USER_CONTEXT[uid].append({"role": "assistant", "content": err})
    try:
        await context.bot.edit_message_text(
            chat_id=placeholder.chat_id,
            message_id=placeholder.message_id,
            text=err,
            disable_web_page_preview=True,
        )
    except Exception:
        await update.message.reply_text(err, disable_web_page_preview=True, reply_markup=main_menu_keyboard())

# -------------------- запуск --------------------
def main() -> None:
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    log.info("Бот запущен (long polling). Ctrl+C — остановка.")
    app.run_polling()

if __name__ == "__main__":
    main()
