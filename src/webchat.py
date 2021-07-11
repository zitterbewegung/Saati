from flask import Flask, render_template, request, session, make_response

# from chatterbot import ChatBot
# from chatterbot.trainers import ChatterBotCorpusTrainer
from flask import send_file
import uvicorn
from logic import answer_question
import sys, logging, os, uuid

app = Flask("saati")
SECRET_KEY = os.environ["SECRET_KEY"]

# create chatbot
# englishBot = ChatBot("Chatterbot", storage_adapter="chatterbot.storage.SQLStorageAdapter")
# trainer = ChatterBotCorpusTrainer(englishBot)
# trainer.train("chatterbot.corpus.english") #train the chatter bot for english

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
app.logger.addHandler(handler)
app.logger.setLevel(logging.DEBUG)

# define app routes
@app.route("/")
def index():
    # user_identifier = session.get('identifier', uuid.uuid4())
    # resp.set_cookie('userID', session['identifier'])
    return render_template("chatbot.html")


@app.route("/talk")
def talk():
    # user_identifier = session.get('identifier', uuid.uuid4())
    # resp.set_cookie('userID', session['identifier'])
    return render_template("talk.html")


# @app.route('/setcookie', methods = ['POST', 'GET'])
# def setcookie():
#   if request.method == 'POST':
#    user = request.form['nm']

#   resp = make_response(render_template('readcookie.html'))
#   resp.set_cookie('userID', user)

#   return resp


@app.route("/get_image")
def get_image():
    if request.args.get("type") == "1":
        filename = "ok.gif"
    else:
        filename = "error.jpg"
    return send_file(filename, mimetype="image/jpg")


@app.route("/get")
# function for the bot response
def get_bot_response():
    # resp = make_response()
    # if not request.cookies.get('userID'):
    #

    cookieid = request.cookies.get("userID")
    # user_identifier = session.get('identifier', uuid.uuid4())

    userText = request.args.get("msg")
    user_identifier = str(request.remote_addr)
    inference = answer_question(userText, user_identifier, "webchat")
    return inference

    # return str(englishBot.get_response(userText))


if __name__ == "__main__":
    app.run(debug=True, port=80, host='0.0.0.0')
