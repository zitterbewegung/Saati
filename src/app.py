#!/usr/bin/env python
# -*- coding: utf-8 -*-
from twilio.rest import Client

from flask import Flask
from flask import request, sessions, jsonify
from flask import render_template
import sys, logging, uuid
import os


import speech_recognition as sr
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from .logic import answer_question, blenderbot400M, blenderbot3B

from twilio.twiml.messaging_response import MessagingResponse
from .translate import Translator
from .config import *


app = Flask(__name__)
app.config["DEBUG"] = True # turn off in prod

translator = Translator(MODEL_PATH)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
app.logger.addHandler(handler)
app.logger.setLevel(logging.DEBUG)

english_bot = ChatBot("Chatterbot", storage_adapter="chatterbot.storage.SQLStorageAdapter")
trainer = ChatterBotCorpusTrainer(english_bot)
trainer.train("chatterbot.corpus.english")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return str(english_bot.get_response(userText))

@app.route('/lang_routes', methods = ["GET"])
def get_lang_route():
    lang = request.args['lang']
    all_langs = translator.get_supported_langs()
    lang_routes = [l for l in all_langs if l[0] == lang]
    return jsonify({"output":lang_routes})

@app.route('/supported_languages', methods=["GET"])
def get_supported_languages():
    langs = translator.get_supported_langs()
    return jsonify({"output":langs})

@app.route('/translate', methods=["POST"])
def get_prediction():
    source = request.json['source']
    target = request.json['target']
    text = request.json['text']
    translation = translator.translate(source, target, text)
    return jsonify({"output":translation})

@app.route('/signUpUser', methods=['POST'])
def signUpUser():
    user =  request.form['username'];
    password = request.form['password'];
    return json.dumps({'status':'OK','user':user,'pass':password})

@app.route("/audio", methods=["POST", "GET"])
def audio_chat():
    if request.method == "POST":

        file = request.files["audio_data"]
        # with open('audio.wav', 'wb') as audio:
        #    f.save(audio)
       
        recognizer = sr.Recognizer()
        audioFile = sr.AudioFile(file)
        
        with audioFile as source:
            data = recognizer.record(source)
        transcript = recognizer.recognize_google(data, key=None)
        print(transcript)
        response = answer_question(transcript)[0]
        transcript = transcript + response
        print("file uploaded successfully")
        return render_template("index2.html", request="POST")
    else:
        return render_template("index2.html")

app.route("/sms", methods=["GET", "POST"])
def sms_reply():
    """Respond to incoming calls with a simple text message."""
    
    # Start our TwiML response
    resp = MessagingResponse()
    account_sid = os.environ["TWILIO_ACCOUNT_SID"]
    auth_token = os.environ["TWILIO_AUTH_TOKEN"]

    client = Client(account_sid, auth_token)

    incoming_msg = request.values.get("Body", None)

    

    responded = False
    if incoming_msg:
        pass
        #Lookup a user
        
        
    DATA_FILENAME = 'state.json'
    if not responded:
        event_log = []
        if os.path.exists("state.json"):
            with open(DATA_FILENAME, mode='r', encoding='utf-8') as feedsjson:
                event_log = json.load(feedsjson)
            #file_pi2 = open('state.json', 'r') 
            #state = file_pi2
        else:
            with open(DATA_FILENAME, mode='w', encoding='utf-8') as f:
                json.dump([], f)

        if event_log != []:
            state = event_log[-1]
        sentiment = state.get('sentiment', 1)
        #sentiment = 1
        interactions = state.get('interactions', 1)
       
        #interactions = 1
        sync_ratio = sentiment / interactions
        responses = state.get('responses', [])

        instance = Saati(uuid.uuid4())
        # instance.get_graph().draw('my_state_diagram.png', prog='dot')
        
        #dump = pickle.dumps(instance)

        # user_input = input #GivenCommand()

        # Add a message

        logging.info("Computing reply")
        resp = MessagingResponse()
        
        #answer_question(incoming_msg)
        responce = blenderbot3B(incoming_msg)[0]
        
        message = client.messages.create(
            body=responce,  # Join Earth's mightiest heroes. Like Kevin Bacon.",
            from_="17784035044",
            to=request.values['From'],
        )
        # Get users phone to respond.
        resp.message(responce)
        # Start our TwiML response

        # talk(responce)
        responses.append(responce)
        sentiment = sentiment + compute_sentiment(incoming_msg)
        interactions = interactions + 1

        logging.info(
            "Responses: {} Sentiment: {}  Sync ratio: {} Interactions: {}	| Current State {}".format(
                str(responses),
                str(sentiment),
                str(sync_ratio),
                str(interactions),
                str(instance.state),
            )
        )

        if 5 >= sync_ratio <= 11 or interactions < 10:

            instance.next_state()
        else:
            talk("Hey, lets stay friends")
            instance.friendzone()
        #file = open('state.pkl', 'wb')
        current_state = {'responses': responses,
                         'sentiment': sentiment,
                         'sync_ratio' : sync_ratio,
                         'interactions': interactions,
                         'instance.state' : instance.state,
                         'request_time':  str(datetime.datetime.now())}
        with open(DATA_FILENAME, mode='w', encoding='utf-8') as feedsjson:
            event_log.append(current_state)
            json.dump(event_log, feedsjson)
            
        return str(responce)



#define app routes
@app.route("/chat_blender")
def chat():
    return render_template("blender.html")

@app.route("/get_blender")
#function for the bot response
def get_bot_response():
    #resp = make_response()
    #if not request.cookies.get('userID'):
    #     #resp.set_cookie('userID', session['identifier'])
    
    cookieid = request.cookies.get('userID')
    user_identifier = session.get('identifier', uuid.uuid4())
   
    userText = request.args.get('msg')
    inference = answer_question(userText, user_identifier)
    #return blenderbot400M(userText)[0]
    return inference

    #return str(englishBot.get_response(userText))



if __name__ == "__main__":
    app.run(debug=True)


