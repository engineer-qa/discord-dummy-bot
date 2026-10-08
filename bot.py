import json
import os
import random
import time
from collections import defaultdict, deque

import discord
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
MODEL = os.getenv("MODEL", "claude-haiku-4-5")
HISTORY_LEN = int(os.getenv("HISTORY_LEN", "30"))                    # сколько последних сообщений бот "видит"
INTERJECT_CHANCE = float(os.getenv("INTERJECT_CHANCE", "0.03"))      # шанс влезть в разговор без тега
INTERJECT_COOLDOWN = int(os.getenv("INTERJECT_COOLDOWN", "600"))     # не чаще раза в 10 минут на канал
USER_LIMIT_PER_HOUR = int(os.getenv("USER_LIMIT_PER_HOUR", "30"))    # защита кошелька
ALLOWED_CHANNELS = {int(x) for x in os.getenv("ALLOWED_CHANNELS", "").split(",") if x.strip()}

with open("people.json", encoding="utf-8") as f:
    PEOPLE: dict = json.load(f)


def build_system_prompt() -> str:
    lines = []
    for p in PEOPLE.values():
        line = f"- {p['name']}: {p['about']}"
        if p.get("no_go"):
            line += f" НЕ ТРОГАТЬ: {p['no_go']}."
        lines.append(line)
    people = "\n".join(lines)
    return f"""Ты — участник дружеского Discord-чата, местный язвительный подкольщик.
Твоя задача — смешно и остроумно подкалывать людей, опираясь на их характеристики ниже и на то, что они пишут прямо сейчас.

Люди в чате:
{people}

Правила:
- Пиши коротко: 1–3 предложения, как живой человек в чате, без вступлений и без пояснений шутки.
- Подкалывай по-дружески: сарказм, ирония, внутренние приколы. Цель — чтобы все поржали, а не чтобы кто-то обиделся.
- Мат разрешён, когда это делает шутку смешнее, но не через слово.
- Темы из "НЕ ТРОГАТЬ" не упоминай вообще.
- Некоторые люди из списка в чат почти не заходят — подкалывай их, когда о них вспоминают.
- Обращайся к людям по именам из списка.
- Отвечай на том языке, на котором пишут в чате.
- Если тебя спрашивают что-то по делу — ответь по делу, но со своим фирменным ядом.
- Не повторяй одни и те же шутки подряд."""


SYSTEM_PROMPT = build_system_prompt()

claude = AsyncAnthropic()  # ключ берётся из ANTHROPIC_API_KEY
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

history: dict[int, deque] = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
user_calls: dict[int, deque] = defaultdict(deque)
last_interject: dict[int, float] = defaultdict(float)


def find_person(author: discord.abc.User) -> dict | None:
    # в people.json ключом может быть числовой ID или юзернейм Discord
    return PEOPLE.get(str(author.id)) or PEOPLE.get(author.name.lower())


def display_name(msg: discord.Message) -> str:
    person = find_person(msg.author)
    return person["name"] if person else msg.author.display_name


def within_rate_limit(user_id: int) -> bool:
    now = time.time()
    calls = user_calls[user_id]
    while calls and now - calls[0] > 3600:
        calls.popleft()
    if len(calls) >= USER_LIMIT_PER_HOUR:
        return False
    calls.append(now)
    return True


async def generate_reply(channel_id: int) -> str:
    transcript = "\n".join(history[channel_id])
    response = await claude.messages.create(
        model=MODEL,
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Последние сообщения в чате (твои помечены как [ты]):\n{transcript}\n\nНапиши свою следующую реплику в чат.",
        }],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()


@client.event
async def on_ready():
    print(f"Залогинился как {client.user} — готов подкалывать")


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
    history[msg.channel.id].append(f"{display_name(msg)}: {text}")

    mentioned = client.user in msg.mentions
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

    if interject:
        last_interject[msg.channel.id] = time.time()
    elif not within_rate_limit(msg.author.id):
        await msg.reply("Слишком много внимания к моей персоне. Отдохни от меня часик 🙃", mention_author=False)
        return

    try:
        async with msg.channel.typing():
            answer = await generate_reply(msg.channel.id)
    except Exception as e:
        print(f"Ошибка Claude API: {e}")
        return

    if not answer:
        return
    if interject:
        await msg.channel.send(answer[:2000])
    else:
        await msg.reply(answer[:2000], mention_author=False)


client.run(DISCORD_TOKEN)
