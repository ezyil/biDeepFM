import telepot

def send(text, logger=False):
    if logger:
        token = """put your token here"""
        chat_id = -1 # put your chat id here
    else:
        token = """put your token here"""
        chat_id = -1 # put your chat id here
        
    bot = telepot.Bot(token)
    bot.sendMessage(chat_id, text)
