#!/usr/bin/python3
#apt install libffi-dev libnacl-dev python3-dev
from discord.ext import commands
import discord
import logging
import os


intents = discord.Intents.default()
intents.members = True

bot = commands.Bot(command_prefix='?', description=description, intents=intents)

@bot.event
async def on_ready():
    print('Logged in as')
    print(bot.user.name)
    print(bot.user.id)
    print('------')

@bot.event
async def on_message(self, message):
    # we do not want the bot to reply to itself
    if message.author.id == self.user.id:
        return

    if message.content.startswith('!hello'):
        await message.reply('Hello!', mention_author=True)

bot.run(os.environ['DISCORD_API_TOKEN'])
