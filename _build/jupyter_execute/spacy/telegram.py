Your first chatbot
==================

A typical chatbot app consists of multiple tiers. After you've implemented
the logic for processing user input on your machine, you'll need a messenger
app (e.g. Skype, Facebook Messenger or Telegram) that allows you to create accounts that your programs operate. Users won't interact with the bot implementation on your machine directly; instead, they'll chat with the bot through the messenger API. Apart from a messenger, your chatbot might require some additional services, such as a database or other storage.

## Creating a Telegram Account and Authorizing Your Bot

```{note}
You'll need a smartphone or tablet that runs either iOS or Android to create a Telegram account. A PC version of Telegram won't work for this operation. However,
once you create a Telegram account, you can use it on a PC.
```
On your smartphone or tablet go to an app store, and install the `Telegram app`. Then you need to enter your `phone number` to access the app. 

1. In the Telegram app, perform a search for @BotFather (note: case sensitive) or open the URL https://telegram.me/botfather/. BotFather is a Telegram bot that manages all the other bots in your account.
2. On the BotFather page, click the Start button to see the list of commands
that you can use to set up your Telegram bots.
3. To create a new bot, enter the `/newbot` command in the Write a message
box. You'll be prompted for a name and a username for your bot. Then
you'll be given an authorization token for the new bot. Note down the token in a secure place as people who have access to the token can manipulate your new bot.

## Install the `python-telegram-bot` Library

To connect chatbot functionality implemented in Python, you'll need the
python-telegram-bot library, which is built on top of the Telegram Bot API.
The library provides an easy-to-use interface for bot programmers developing
apps for Telegram.

```
pip install python-telegram-bot --upgrade
```
Once you've installed the library, use the following lines of code to
perform a quick test to verify that you can access your Telegram bot from
Python. You must have an internet connection for this test to work.

import telegram
TOKEN = 'YOUR TOKEN'
bot = telegram.Bot(token=TOKEN)

print(bot.get_me())

## A copy-cat chatbot
A chatbot simply echoing the input message. 

from telegram.ext import Updater, MessageHandler, Filters
#function that implements the message handler
def echo(update, context):
    update.message.reply_text(update.message.text)
#creating an Updater instance
updater = Updater(TOKEN, use_context=True)
#registering a handler to handle input text messages
updater.dispatcher.add_handler(MessageHandler(Filters.text, echo))
#starting polling updates from the messenger
updater.start_polling()
updater.idle()

Click the `run` button for the above cell, it will start the bot. You can then go to your smartphone or tablet's Telegram app to chat with your bot. 

```{note}
You can use the Interrupt button in your notebook to stop the bot after testing.
```

## Chatbot that uses spaCy code

import spacy
from telegram.ext import Updater, MessageHandler, Filters
#the callback function that uses spaCy
def utterance(update, context):
    msg = update.message.text
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(msg)
    for token in doc:
        if token.dep_ == 'dobj':
            update.message.reply_text('We are processing your request...')
            return 
    update.message.reply_text('Please rephrase your request. Be as specific as possible!')

#the code responsible for interactions with Telegram
updater = Updater(TOKEN, use_context=True)
updater.dispatcher.add_handler(MessageHandler(Filters.text, utterance))
updater.start_polling()
updater.idle()

```{admonition} Your Turn
Integrte some more patterns to handle more varieties of requests. For example, extracting the type of pizza by finding its left children, if you are handling pizza ordering. 
```
**Reference**: Chapter 11 of NATURAL LANGUAGE PROCESSING WITH PYTHON AND SPACY