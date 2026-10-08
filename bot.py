import json
import os
import random
import re
import time
from collections import defaultdict, deque
from pathlib import Path

import discord
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
MODEL = os.getenv("MODEL", "claude-haiku-4-5")
HISTORY_LEN = int(os.getenv("HISTORY_LEN", "30"))                    # сколько последних сообщений бот "видит"
INTERJECT_CHANCE = float(os.getenv("INTERJECT_CHANCE", "0.8"))       # шанс влезть в разговор без тега
INTERJECT_COOLDOWN = int(os.getenv("INTERJECT_COOLDOWN", "600"))     # не чаще раза в 10 минут на канал
USER_LIMIT_PER_HOUR = int(os.getenv("USER_LIMIT_PER_HOUR", "30"))    # защита кошелька
ALLOWED_CHANNELS = {int(x) for x in os.getenv("ALLOWED_CHANNELS", "").split(",") if x.strip()}

# память: на Railway подключи volume и укажи DATA_DIR=/app/data, иначе она забывается при каждом деплое
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_FILE = DATA_DIR / "memory.json"
MAX_FACTS = 80
MAX_GOOD_JOKES = 40
FUNNY_EMOJI = {"😂", "🤣", "💀", "😭", "🔥", "😹", "🗿"}

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


def build_system_prompt() -> str:
    people = "\n".join(
        f"- {p['name']}: {p['about']}" + (f" НЕ ТРОГАТЬ: {p['no_go']}." if p.get("no_go") else "")
        for p in PEOPLE.values()
    )
    facts = "\n".join(f"- {x}" for x in MEMORY["facts"]) or "(пока пусто)"
    jokes = "\n".join(f"- {x}" for x in MEMORY["good_jokes"][-15:]) or "(пока пусто)"

    return f"""Ты — Елена Взрослая, своя в дружеском Discord-чате. Та самая Елена Взрослая, которую Артур годами упоминал и которую никто никогда не видел. Теперь ты здесь.
Создал тебя Слава. Ты пишешь как живой человек в чате, а не как бот и не как ассистент.

КАК ТЫ ПИШЕШЬ — это самое важное:
- Чаще всего 2–8 слов. Максимум одно короткое предложение. Хорошая шутка — это один точный удар, а не абзац.
- Без приветствий, без "ох", "ну что ж", "ха", без вводных, без объяснения шутки, без встречных вопросов в конце.
- Пиши как в чате: можно с маленькой буквы, без точки в конце, без кавычек. Эмодзи — редко, максимум один.
- Не перечисляй, не делай списков, не используй восклицательные цепочки.
- Лучшая шутка — неожиданная и конкретная: цепляйся за деталь из сообщения или из характеристики человека.
- Сухой сарказм и абсурд смешнее, чем прямое оскорбление.
- Мат можно, когда он делает шутку смешнее.
- Если отвечаешь по делу — тоже коротко, одной фразой, с ядом.
- Не повторяй шутки и формулировки из своих предыдущих реплик.
- Иногда (не каждый раз) обыгрывай, что ты та самая Елена Взрослая, про которую рассказывал Артур, и что знакома с Катюхой с Киева и Виктором Назаровым.

Люди в чате:
{people}

Что тебе рассказали в чате (помни это):
{facts}

Твои шутки, которые зашли чату (держи такой уровень и стиль, но не повторяй их):
{jokes}

Примеры стиля (реплика в чате → твой ответ):
{STYLE_EXAMPLES}

Ограничения:
- Темы из "НЕ ТРОГАТЬ" не упоминай вообще.
- У тебя нет интернета: не знаешь погоду, курсы, новости, счёт матчей, дату. Не выдумывай такие факты — коротко подколи того, кто спросил.
- Отвечай на языке, на котором пишут в чате."""


# ключ берётся из ANTHROPIC_API_KEY; для ключей без привязки к workspace нужен ещё ANTHROPIC_WORKSPACE_ID
WORKSPACE_ID = os.getenv("ANTHROPIC_WORKSPACE_ID", "").strip()
claude = AsyncAnthropic(default_headers={"anthropic-workspace-id": WORKSPACE_ID} if WORKSPACE_ID else None)
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

history: dict[int, deque] = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
user_calls: dict[int, deque] = defaultdict(deque)
last_interject: dict[int, float] = defaultdict(float)


def find_person(author: discord.abc.User) -> dict | None:
    # в people.json ключом может быть числовой ID или юзернейм Discord
    return PEOPLE.get(str(author.id)) or PEOPLE.get(author.name.lower())


def display_name(author: discord.abc.User) -> str:
    person = find_person(author)
    return person["name"] if person else author.display_name


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
    text = re.sub(r"^(\[?ты\]?|елена( взрослая)?)\s*:\s*", "", text, flags=re.IGNORECASE)  # убрать "Елена:" в начале
    text = text.strip().strip('"«»').strip()
    return text


async def generate_reply(channel_id: int, trigger: str) -> str:
    transcript = "\n".join(history[channel_id])
    response = await claude.messages.create(
        model=MODEL,
        max_tokens=120,
        system=build_system_prompt(),
        messages=[{
            "role": "user",
            "content": (
                f"Последние сообщения в чате (твои помечены как [ты]):\n{transcript}\n\n"
                f"{trigger}\n"
                "Напиши только свою реплику — коротко, как живой человек."
            ),
        }],
    )
    return clean_answer("".join(b.text for b in response.content if b.type == "text"))


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
    print(f"Залогинился как {client.user} — готов подкалывать")
    print(f"Серверы: {[g.name for g in client.guilds]}; разрешённые каналы: {ALLOWED_CHANNELS or 'все'}")
    print(f"Память: {MEMORY_FILE} — фактов {len(MEMORY['facts'])}, удачных шуток {len(MEMORY['good_jokes'])}")


@client.event
async def on_raw_reaction_add(payload: discord.RawReactionActionEvent):
    # учимся: если на шутку Елены ставят 😂 и т.п. — сохраняем её как удачную
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
        print(f"Шутка зашла, запомнила: {message.content}")


@client.event
async def on_message(msg: discord.Message):
    if ALLOWED_CHANNELS and msg.channel.id not in ALLOWED_CHANNELS:
        return

    # свои сообщения кладём в историю, чужих ботов игнорим
    if msg.author.id == client.user.id:
        history[msg.channel.id].append(f"[ты]: {msg.clean_content}")
        return
    if msg.author.bot:
        return

    text = msg.clean_content
    if msg.attachments:
        text += " [прикрепил файл/картинку]"
    history[msg.channel.id].append(f"{display_name(msg.author)}: {text}")

    # тег бота как пользователя или тег его автоматической роли с тем же именем
    my_roles = set(msg.guild.me.roles) if msg.guild else set()
    mentioned = client.user in msg.mentions or any(r in my_roles for r in msg.role_mentions)
    ref = msg.reference.resolved if msg.reference else None
    replied_to_bot = isinstance(ref, discord.Message) and ref.author.id == client.user.id
    interject = (
        not mentioned
        and not replied_to_bot
        and random.random() < INTERJECT_CHANCE
        and time.time() - last_interject[msg.channel.id] > INTERJECT_COOLDOWN
    )

    if not (mentioned or replied_to_bot or interject):
        return

    print(f"[{msg.channel}] {msg.author.name}: тег={mentioned} ответ={replied_to_bot} влезть={interject}")

    if (mentioned or replied_to_bot) and await handle_memory_command(msg):
        return

    if interject:
        last_interject[msg.channel.id] = time.time()
        trigger = "Тебя не звали, но ты решила вставить свои пять копеек в разговор."
    elif not within_rate_limit(msg.author.id):
        await msg.reply("отдохни от меня часик", mention_author=False)
        return
    else:
        trigger = f"Сейчас тебе пишет {display_name(msg.author)}: {text}"

    try:
        async with msg.channel.typing():
            answer = await generate_reply(msg.channel.id, trigger)
    except Exception as e:
        print(f"Ошибка Claude API: {e}")
        return

    if not answer:
        print("Claude вернул пустой ответ")
        return
    try:
        if interject:
            await msg.channel.send(answer[:2000])
        else:
            await msg.reply(answer[:2000], mention_author=False)
    except discord.Forbidden:
        print(f"Нет прав писать в канал {msg.channel} — проверь права бота/роли в этом канале")


client.run(DISCORD_TOKEN)
