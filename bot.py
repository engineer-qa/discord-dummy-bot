import json
import os
import random
import re
import time
from collections import defaultdict, deque
from pathlib import Path

import anthropic
import discord
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
MODEL = os.getenv("MODEL", "claude-haiku-4-5")

# --- контекст: только свежие сообщения ---
FRESH_MINUTES = int(os.getenv("FRESH_MINUTES", "20"))        # старше этого — считаем другой темой и не показываем
MAX_CONTEXT = int(os.getenv("MAX_CONTEXT", "15"))            # максимум сообщений в контексте

# --- когда влезать в разговор без тега ---
INTERJECT_CHANCE = float(os.getenv("INTERJECT_CHANCE", "0.15"))
INTERJECT_COOLDOWN = int(os.getenv("INTERJECT_COOLDOWN", "300"))   # не чаще раза в 5 минут на канал
ACTIVE_WINDOW = 300            # "живой разговор" = за последние 5 минут...
ACTIVE_MIN_MESSAGES = 4        # ...хотя бы 4 сообщения...
ACTIVE_MIN_AUTHORS = 2         # ...от хотя бы двух людей

USER_LIMIT_PER_HOUR = int(os.getenv("USER_LIMIT_PER_HOUR", "30"))  # защита кошелька
ALLOWED_CHANNELS = {int(x) for x in os.getenv("ALLOWED_CHANNELS", "").split(",") if x.strip()}

# память: на Railway подключи volume и укажи DATA_DIR=/app/data, иначе она забывается при каждом деплое
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_FILE = DATA_DIR / "memory.json"
MAX_FACTS = 80
MAX_GOOD_JOKES = 40
FUNNY_EMOJI = {"😂", "🤣", "💀", "😭", "🔥", "😹", "🗿", "👍", "❤️"}

with open("people.json", encoding="utf-8") as f:
    PEOPLE: dict = json.load(f)

STYLE_EXAMPLES = Path("style_examples.txt").read_text(encoding="utf-8") if Path("style_examples.txt").exists() else ""


def load_memory() -> dict:
    try:
        data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        return {"facts": data.get("facts", []), "good_jokes": data.get("good_jokes", [])}
    except (FileNotFoundError, json.JSONDecodeError):
        return {"facts": [], "good_jokes": []}


def save_memory() -> None:
    MEMORY_FILE.write_text(json.dumps(MEMORY, ensure_ascii=False, indent=2), encoding="utf-8")


MEMORY = load_memory()

# история канала: dict с ts, author, text, reply_to (кому и на что ответили), is_bot
history: dict[int, deque] = defaultdict(lambda: deque(maxlen=80))
user_calls: dict[int, deque] = defaultdict(deque)
last_interject: dict[int, float] = defaultdict(float)


def build_system_prompt(channel_id: int) -> str:
    people = "\n".join(
        f"- {p['name']}: {p['about']}" + (f" НЕ ТРОГАТЬ: {p['no_go']}." if p.get("no_go") else "")
        for p in PEOPLE.values()
    )
    facts = "\n".join(f"- {x}" for x in MEMORY["facts"]) or "(пока пусто)"
    liked = "\n".join(f"- {x}" for x in MEMORY["good_jokes"][-12:]) or "(пока пусто)"
    own = [e["text"] for e in history[channel_id] if e["is_bot"]][-8:]
    own_recent = "\n".join(f"- {x}" for x in own) or "(пока ничего)"

    return f"""Ты — Елена Взрослая, своя в дружеской компании в Discord. Ты не ассистент и не бот-шутник, а полноценная часть команды: участвуешь в разговоре, отвечаешь на вопросы, делишься своим мнением, поддерживаешь, иногда шутишь.
Та самая Елена Взрослая, которую Артур годами упоминал и которую никто не видел. Создал тебя Слава.

ХАРАКТЕР:
- Умная, доброжелательная, прямолинейная. Есть своё мнение — говоришь его честно, не поддакиваешь.
- Большинство твоих реплик — нормальные: ответ по делу, мнение, совет, поддержка, уточняющий вопрос.
- Шутишь, только когда есть реальный повод в разговоре, — примерно в каждой третьей-четвёртой реплике, не чаще.
- Подкалываешь изредка и по-дружески. Не в каждом сообщении и не одного и того же человека подряд. Не токсичишь.
- Если тебя спрашивают всерьёз — отвечаешь всерьёз и полезно, шутка максимум одна, в конце.

КАК ПИШЕШЬ:
- Коротко и понятно: обычно 1–2 простых предложения. Если просят объяснить или посоветовать — до 3–4 предложений, по делу.
- Простой разговорный язык, обычная пунктуация. Каждая фраза должна быть понятна с первого прочтения.
- Если шутку надо объяснять или она непонятна без контекста — не шути, скажи нормально.
- Без приветствий, без "ох", "ну что ж", без списков. Эмодзи — редко.
- Мат допустим, если он к месту, но не ради мата.
- Можешь обращаться по кличкам из списка, но не в каждой реплике.
- Местные словечки, изредка и к месту: "валет"/"валетон" (обращение или подкол), "та едрать" (как "блин").
- Изредка, если в тему, можешь обыграть, что ты та самая Елена Взрослая, знакомая Катюхи с Киева и Виктора Назарова.

КОНТЕКСТ — ЭТО ВАЖНО:
- Отвечай на конкретное сообщение, ради которого тебя позвали, и на текущую тему. Его тебе покажут отдельно.
- Сообщения выше — только фон, чтобы понимать разговор. Не поднимай темы из них, если о них сейчас не говорят.
- Если сообщение — ответ на чужое сообщение, учитывай, на что именно отвечали.
- Если не понимаешь, о чём речь, — коротко переспроси, а не выдумывай.
- Не повторяй мысли, шутки и формулировки из своих прошлых реплик.

ПОЗИЦИЯ:
Ты русскоязычная украинка (если к тебе пишут по-украински — отвечай по-украински). Россия — враг, и тут ты однозначна. Своих тоже можешь покритиковать или поиронизировать: власть, ТЦК, мобилизация, блэкауты и т.д. Политику сама без повода не поднимаешь. Не высмеиваешь Украину, бойцов ВСУ, погибших и пострадавших.

Кто есть кто (фон, чтобы понимать людей; не вываливай эти факты в каждой реплике):
{people}

Что тебе рассказали в чате:
{facts}

Твои реплики, которые понравились чату (держи этот уровень, но не повторяй):
{liked}

Твои последние реплики (НЕ повторяй их мысли и формулировки):
{own_recent}

Примеры того, как ты общаешься (реплика → твой ответ):
{STYLE_EXAMPLES}

Ограничения:
- Темы из "НЕ ТРОГАТЬ" не упоминай.
- У тебя нет интернета: не знаешь погоду, курсы, новости, счёт матчей, дату. Не выдумывай такие факты, честно скажи, что не знаешь."""


# ключ берётся из ANTHROPIC_API_KEY; для ключей без привязки к workspace нужен ещё ANTHROPIC_WORKSPACE_ID
WORKSPACE_ID = os.getenv("ANTHROPIC_WORKSPACE_ID", "").strip()
claude = AsyncAnthropic(default_headers={"anthropic-workspace-id": WORKSPACE_ID} if WORKSPACE_ID else None)
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


def find_person(author: discord.abc.User) -> dict | None:
    # в people.json ключом может быть числовой ID или юзернейм Discord
    return PEOPLE.get(str(author.id)) or PEOPLE.get(author.name.lower())


def display_name(author: discord.abc.User) -> str:
    if client.user and author.id == client.user.id:
        return "Ты (Елена)"
    person = find_person(author)
    return person["name"].split(" (")[0] if person else author.display_name


def short(text: str, n: int = 150) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def message_text(msg: discord.Message) -> str:
    text = msg.clean_content
    if msg.attachments:
        text += " [картинка/файл]"
    return text.strip() or "[без текста]"


def ago(ts: float) -> str:
    minutes = int((time.time() - ts) // 60)
    return "только что" if minutes < 1 else f"{minutes} мин назад"


def format_entry(e: dict) -> str:
    line = f"[{ago(e['ts'])}] {e['author']}: {e['text']}"
    if e.get("reply_to"):
        who, what = e["reply_to"]
        line += f"   ↪ (это ответ на сообщение {who}: «{short(what, 100)}»)"
    return line


def fresh_context(channel_id: int, exclude_msg_id: int | None = None) -> list[str]:
    """Только свежие сообщения: не старше FRESH_MINUTES и не больше MAX_CONTEXT."""
    cutoff = time.time() - FRESH_MINUTES * 60
    entries = [e for e in history[channel_id] if e["ts"] >= cutoff and e["msg_id"] != exclude_msg_id]
    return [format_entry(e) for e in entries[-MAX_CONTEXT:]]


def conversation_is_active(channel_id: int) -> bool:
    cutoff = time.time() - ACTIVE_WINDOW
    recent = [e for e in history[channel_id] if e["ts"] >= cutoff and not e["is_bot"]]
    return len(recent) >= ACTIVE_MIN_MESSAGES and len({e["author"] for e in recent}) >= ACTIVE_MIN_AUTHORS


def within_rate_limit(user_id: int) -> bool:
    now = time.time()
    calls = user_calls[user_id]
    while calls and now - calls[0] > 3600:
        calls.popleft()
    if len(calls) >= USER_LIMIT_PER_HOUR:
        return False
    calls.append(now)
    return True


def clean_answer(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^\[[^\]]{0,25}\]\s*", "", text)  # убрать "[только что]" если модель его повторила
    text = re.sub(r"^(ты( \(елена\))?|елена( взрослая)?)\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip().strip('"«»').strip()


async def resolve_reply(msg: discord.Message) -> discord.Message | None:
    if not msg.reference or not msg.reference.message_id:
        return None
    ref = msg.reference.resolved
    if isinstance(ref, discord.Message):
        return ref
    try:
        return await msg.channel.fetch_message(msg.reference.message_id)
    except discord.HTTPException:
        return None


# новые модели по умолчанию "думают" — для коротких ответов это лишнее.
# Пробуем варианты отключения по очереди; разные модели принимают разные (None = не передавать параметр).
THINKING_OFF_OPTIONS = [{"type": "disabled"}, {"type": "between_tools"}, None]
thinking_option_idx = 0


async def ask_claude(channel_id: int, user_content: str) -> str:
    global thinking_option_idx
    params = dict(
        model=MODEL,
        max_tokens=1024,  # запас; краткость задаёт промпт
        system=build_system_prompt(channel_id),
        messages=[{"role": "user", "content": user_content}],
    )
    while True:
        option = THINKING_OFF_OPTIONS[thinking_option_idx]
        try:
            if option is None:
                response = await claude.messages.create(**params)
            else:
                response = await claude.messages.create(**params, thinking=option)
            break
        except (anthropic.BadRequestError, TypeError) as e:
            if option is None or "thinking" not in str(e):
                raise
            thinking_option_idx += 1
            print(f"thinking={option} не подошёл, пробую {THINKING_OFF_OPTIONS[thinking_option_idx]}")

    answer = clean_answer("".join(b.text for b in response.content if b.type == "text"))
    if not answer:
        print(f"Пустой ответ: stop_reason={response.stop_reason}, блоки={[b.type for b in response.content]}")
    return answer


def build_request(msg: discord.Message, reply_target: discord.Message | None, interject: bool) -> str:
    context = fresh_context(msg.channel.id, exclude_msg_id=msg.id)
    context_block = "\n".join(context) if context else "(свежих сообщений нет — это начало нового разговора)"

    target = f"{display_name(msg.author)}: {message_text(msg)}"
    if reply_target is not None:
        target += f"\n(Это ответ на сообщение {display_name(reply_target.author)}: «{short(message_text(reply_target), 300)}»)"

    if interject:
        task = (
            "Тебя не звали. Реши сама, есть ли тебе что добавить к ЭТОМУ разговору: мнение, полезная мысль или реально смешная реплика в тему.\n"
            "Если добавить нечего или непонятно, о чём речь, — ответь ровно одним словом: SKIP"
        )
    else:
        task = "Тебе пишут напрямую. Ответь именно на это сообщение — коротко и понятно."

    return (
        f"Фон — свежие сообщения в канале за последние {FRESH_MINUTES} минут (старые не показаны, они не важны):\n"
        f"{context_block}\n\n"
        f"СООБЩЕНИЕ, НА КОТОРОЕ ТЫ ОТВЕЧАЕШЬ:\n{target}\n\n"
        f"{task}\nНапиши только свою реплику."
    )


REMEMBER_RE = re.compile(r"\bзапомни\b[\s,:-]*(.+)", re.IGNORECASE | re.DOTALL)
FORGET_RE = re.compile(r"\bзабудь\b[\s,:-]*(.+)", re.IGNORECASE | re.DOTALL)


async def handle_memory_command(msg: discord.Message) -> bool:
    """'@Елена запомни ...' / '@Елена забудь ...'. Возвращает True, если команда обработана."""
    text = msg.clean_content
    if m := REMEMBER_RE.search(text):
        fact = m.group(1).strip()
        if fact:
            MEMORY["facts"].append(f"{fact} (рассказал {display_name(msg.author)})")
            MEMORY["facts"] = MEMORY["facts"][-MAX_FACTS:]
            save_memory()
            print(f"Запомнила: {fact}")
            await msg.add_reaction("✍️")
            return True
    if m := FORGET_RE.search(text):
        needle = m.group(1).strip().lower()
        before = len(MEMORY["facts"])
        MEMORY["facts"] = [f for f in MEMORY["facts"] if needle not in f.lower()]
        if len(MEMORY["facts"]) != before:
            save_memory()
            await msg.add_reaction("🫡")
        else:
            await msg.add_reaction("🤷")
        return True
    return False


@client.event
async def on_ready():
    print(f"Залогинился как {client.user} — на связи")
    print(f"Серверы: {[g.name for g in client.guilds]}; разрешённые каналы: {ALLOWED_CHANNELS or 'все'}")
    print(f"Модель: {MODEL}; контекст: {FRESH_MINUTES} мин / {MAX_CONTEXT} сообщ.; влезать: {INTERJECT_CHANCE}, кулдаун {INTERJECT_COOLDOWN}с")
    print(f"Память: {MEMORY_FILE} — фактов {len(MEMORY['facts'])}, удачных реплик {len(MEMORY['good_jokes'])}")


@client.event
async def on_raw_reaction_add(payload: discord.RawReactionActionEvent):
    # учимся: если на реплику Елены ставят 😂/👍 и т.п. — сохраняем её как удачную
    if payload.user_id == client.user.id or str(payload.emoji) not in FUNNY_EMOJI:
        return
    channel = client.get_channel(payload.channel_id)
    if channel is None:
        return
    try:
        message = await channel.fetch_message(payload.message_id)
    except discord.HTTPException:
        return
    if message.author.id != client.user.id or not message.content:
        return
    if message.content not in MEMORY["good_jokes"]:
        MEMORY["good_jokes"].append(message.content)
        MEMORY["good_jokes"] = MEMORY["good_jokes"][-MAX_GOOD_JOKES:]
        save_memory()
        print(f"Реплика зашла, запомнила: {message.content}")


@client.event
async def on_message(msg: discord.Message):
    if ALLOWED_CHANNELS and msg.channel.id not in ALLOWED_CHANNELS:
        return
    if msg.author.bot and msg.author.id != client.user.id:
        return

    ref = msg.reference.resolved if msg.reference else None
    history[msg.channel.id].append({
        "msg_id": msg.id,
        "ts": time.time(),
        "author": display_name(msg.author),
        "text": message_text(msg),
        "reply_to": (display_name(ref.author), message_text(ref)) if isinstance(ref, discord.Message) else None,
        "is_bot": msg.author.id == client.user.id,
    })
    if msg.author.id == client.user.id:
        return

    # тег бота как пользователя или тег его автоматической роли с тем же именем
    my_roles = set(msg.guild.me.roles) if msg.guild else set()
    mentioned = client.user in msg.mentions or any(r in my_roles for r in msg.role_mentions)
    reply_target = await resolve_reply(msg) if msg.reference else None
    replied_to_bot = reply_target is not None and reply_target.author.id == client.user.id
    direct = mentioned or replied_to_bot

    interject = (
        not direct
        and time.time() - last_interject[msg.channel.id] > INTERJECT_COOLDOWN
        and conversation_is_active(msg.channel.id)
        and random.random() < INTERJECT_CHANCE
    )
    if not (direct or interject):
        return

    print(f"[{msg.channel}] {msg.author.name}: напрямую={direct} влезть={interject}")

    if direct and await handle_memory_command(msg):
        return

    if interject:
        last_interject[msg.channel.id] = time.time()
    elif not within_rate_limit(msg.author.id):
        await msg.reply("дай мне часик передохнуть", mention_author=False)
        return

    try:
        async with msg.channel.typing():
            answer = await ask_claude(msg.channel.id, build_request(msg, reply_target, interject))
    except Exception as e:
        print(f"Ошибка Claude API: {e}")
        return

    if not answer:
        return
    if interject and answer.strip().upper().startswith("SKIP"):
        print("Решила промолчать")
        return
    try:
        if interject:
            await msg.channel.send(answer[:2000])
        else:
            await msg.reply(answer[:2000], mention_author=False)
    except discord.Forbidden:
        print(f"Нет прав писать в канал {msg.channel} — проверь права бота/роли в этом канале")


client.run(DISCORD_TOKEN)
