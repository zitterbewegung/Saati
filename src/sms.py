#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from twilio.rest import Client
from flask import Flask, request, redirect
from twilio.twiml.messaging_response import MessagingResponse
from logic import Saati, compute_sentiment
from inference_functions import compute_sentiment, blenderbot400M, blenderbot3B, general_questions, emotion_category, openai_utterance
import uuid, logging, os, pickle, json, datetime
from logic import answer_question
import redis 

logging.getLogger("transitions").setLevel(logging.INFO)
app = Flask(__name__)


"""
	If pos or neg pos 5 to 1 relationship doesn't continue
	If exceeds 11 pos 1 neg no challenge
	you wlant not bliss but
"""


instance = Saati(uuid.uuid4())
r = redis.Redis('localhost', 6379, 0)

# instance.get_graph().draw('my_state_diagram.png', prog='dot')
responses = []
# user_input = input #GivenCommand()

@app.route("/sms", methods=["GET", "POST"])
def sms_reply():
    """Respond to incoming calls with a simple text message."""
    
    
    # Start our TwiML response
    resp = MessagingResponse()
    account_sid = os.environ["TWILIO_ACCOUNT_SID"]
    auth_token = os.environ["TWILIO_AUTH_TOKEN"]

    client = Client(account_sid, auth_token)

    incoming_msg = request.values.get("Body", None)

    responded = False
    #if incoming_msg:
    #    pass
    #    # Lookup a user

    DATA_FILENAME = "{}-state.json".format(request.values["From"])
    if not responded:
        event_log = []
        if os.path.exists(DATA_FILENAME):
            with open(DATA_FILENAME, mode="r", encoding="utf-8") as feedsjson:
                event_log = json.load(feedsjson)

        else:
            with open(DATA_FILENAME, mode="w", encoding="utf-8") as f:
                json.dump([], f)
        state = {}
        
        if event_log != []:
            state = event_log[-1]
        sentiment = state.get("sentiment", 0)
        #sync_ratio = state.get("sync_ratio" , 1)
        interactions = state.get("interactions", 1)
        positive_interactions = state.get("positive_interactions", 1)
        negative_interactions = state.get("negative_interactions", 1)
        # interactions = 1
     
        level_counter = state.get("level_counter", 1)
        responses = state.get("responses", [])
        
        # instance.get_graph().draw('my_state_diagram.png', prog='dot')

        # dump = pickle.dumps(instance)

        # user_input = input #GivenCommand()

        # Add a message

        logging.info("Computing reply")
        resp = MessagingResponse()
        
        sentiment = compute_sentiment(incoming_msg)
       
        if sentiment == 'POSITIVE':
            positive_interactions = positive_interactions + 1
        if sentiment == 'NEGATIVE':
            negative_interactions = negative_interactions + 1

        
        
        # answer_question(incoming_msg)
        #if incoming_msg == 'What did we talk about?':
        #    previous_interactions = 
        #    raw_responce = general_questions(

        #
        logging.info('Start inference')
         
        responce = blenderbot3B(incoming_msg)[0]

        #responce = openai_utterance(incoming_msg)
        
        logging.info("Raw respnce: {}".format(responce))
        #responce = raw_responce
        
        
        message = client.messages.create(
            body=responce,  # Join Earth's mightiest heroes. Like Kevin Bacon.",
            from_="17784035044",
            to=request.values["From"],
        )
        # Get users phone to respond.
        resp.message(responce)
        # Start our TwiML response
        interactions = interactions + 1
        
        sync_ratio = positive_interactions / negative_interactions

        if sync_ratio > 5 and sync_ratio < 11: 
            level_counter = level_counter + 1  
            #instance.next_state()
        if sync_ratio > 11 or sync_ratio < 5:
            level_counter = level_counter - 1   

    
        state_message = client.messages.create(
            body="Sentiment: {}  Sync ratio: {} Level_counter {} Positive interactions {} Negative interactions {}	| Current State {}".format(
                #str(responses),
                str(sentiment),
                str(sync_ratio),
                str(level_counter), 
                str(positive_interactions),
                str(negative_interactions),
                instance.state,
            ),  # Join Earth's mightiest heroes. Like Kevin Bacon.",
            from_="17784035044",
            to=request.values["From"],
        )


        # talk(responce)
        responses.append(responce)
        #sentiment = sentiment + 

        logging.info(
            "Incoming Message: {} Responses: {} Sentiment: {}  Sync ratio: {} Interactions: {} Positive Interactions {} Negative Interactions {} level_counter {} Current State {}, response_sentiment {} request_time {}".format(
                incoming_msg,
                responses,
                sentiment,
                sync_ratio,
                interactions,
                positive_interactions,
                negative_interactions,
                level_counter,
                instance.state,
                emotion_category(responce),
                str(datetime.datetime.now()),
                request.values["From"],
                "sms",
            )
        )
        current_state = {
            "responses": responses,
            "sentiment": sentiment,
            "sync_ratio": sync_ratio,
            "incoming_msg" : incoming_msg,
            "interactions": interactions,
            
            "positive_interactions": positive_interactions,
            "negative_interactions": negative_interactions,

            "level_counter" : level_counter,
            "response_sentiment": emotion_category(responce),
            "request_time": str(datetime.datetime.now()),
            "identifier": request.values["From"],
            "origin": "sms",
        }

            #  interactions > 5 and (sync_ratio < 5 or sync_ratio > 11):
            #responce = "Hey, lets stay friends"
            #instance.friendzone()
        # file = open('state.pkl', 'wb')
        # with engine.begin() as connection:
        #    state_df = pd.DataFrame({"identifier" : identifier, 'response': response, 'sentiment': sentiment, "sync_ratio": sync_ratio, "interactions": interactions, "request": body, "identifier": identifier, "origin": origin})
        #    state_df.to_sql('interactions', con=connection, if_exists='append')
        #    log.debug("Current state: {}".format(event_log))
       
        with open(DATA_FILENAME, mode="w", encoding="utf-8") as feedsjson:
            event_log.append(current_state)
            json.dump(event_log, feedsjson)
        logging.debug(request.values["From"])
        r.mset({request.values["From"] : str(event_log)})     
        return str(responce)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
    #print(message.sid)
