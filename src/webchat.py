from flask import Flask, render_template, request, session, make_response
#from chatterbot import ChatBot
#from chatterbot.trainers import ChatterBotCorpusTrainer
from logic import answer_question
import sys, logging, uuid

app = Flask('saati')
app.secret_key = b'34-1q98ghb3q4tg89u'

#create chatbot
#englishBot = ChatBot("Chatterbot", storage_adapter="chatterbot.storage.SQLStorageAdapter")
#trainer = ChatterBotCorpusTrainer(englishBot)
#trainer.train("chatterbot.corpus.english") #train the chatter bot for english

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
app.logger.addHandler(handler)
app.logger.setLevel(logging.DEBUG)

#define app routes
@app.route("/chatbot")
def index():
    return render_template("chatbot.html")

@app.route("/get")
#function for the bot response
def get_bot_response():
    #resp = make_response()
    if not request.cookies.get('userID'):
        resp.set_cookie('userID', session['identifier'])
    
    cookieid = request.cookies.get('userID')
    user_identifier = session.get('identifier', uuid.uuid4())

    userText = request.args.get('msg')
    inference = answer_question(userText, user_identifier, 'webchat')
    return inference

    #return str(englishBot.get_response(userText))

if __name__ == "__main__":
    app.run(debug=True)
