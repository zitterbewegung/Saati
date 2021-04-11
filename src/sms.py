#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from twilio.rest import Client
from flask import Flask, request, redirect
from twilio.twiml.messaging_response import MessagingResponse
from logic import Saati#, compute_sentiment
from inference_functions import compute_sentiment, blenderbot400M, blenderbot1B
import uuid, logging, os, pickle, json, datetime
from logic import answer_question

logging.getLogger("transitions").setLevel(logging.INFO)
app = Flask(__name__)


"""
	If pos or neg pos 5 to 1 relationship doesn't continue
	If exceeds 11 pos 1 neg no challenge
	you wlant not bliss but
"""


instance = Saati(uuid.uuid4())


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
    if incoming_msg:
        pass
        #Lookup a user
        
      
    DATA_FILENAME = '{}-state.json'.format(request.values['From'])
    if not responded:
        event_log = []
        if os.path.exists(DATA_FILENAME):
            with open(DATA_FILENAME, mode='r', encoding='utf-8') as feedsjson:
                event_log = json.load(feedsjson)
                
        else:
            with open(DATA_FILENAME, mode='w', encoding='utf-8') as f:
                json.dump([], f)
        state = {}
        if event_log != []:
            state = event_log[-1]
        sentiment = state.get('sentiment', 1)
     
        interactions = state.get('interactions', 1)
        positive_interactions = state.get('positive_interactions', 1)
       
        #interactions = 1
        if sentiment > 0:
            positive_interactions = positive_interactions + 1
        else:
            positive_interactions = positive_interactions - 1

        sync_ratio = positive_interactions / interactions
        responses = state.get('responses', [])

        instance = Saati(uuid.uuid4())
        # instance.get_graph().draw('my_state_diagram.png', prog='dot')
        
        #dump = pickle.dumps(instance)

        # user_input = input #GivenCommand()

        # Add a message

        logging.info("Computing reply")
        resp = MessagingResponse()
        
        #answer_question(incoming_msg)
        responce = blenderbot400M(incoming_msg)[0]
        #responce = blenderbot1B(incoming_msg)[0]
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
        current_state = {'responses': responses,
                    'sentiment': sentiment,
                    'sync_ratio' : sync_ratio,
                    'interactions': interactions,
                    'positive_interactions': positive_interactions,
                    'instance.state' : instance.state,
                    'request_time':  str(datetime.datetime.now()),
                    'identifier' : request.values['From'],
                    'origin' : "sms"
                    }

        if 5 >= sync_ratio <= 11 or interactions < 10:
            
            instance.next_state()
        else:
            responce = "Hey, lets stay friends"
            instance.friendzone()
        #file = open('state.pkl', 'wb')
        #with engine.begin() as connection:
        #    state_df = pd.DataFrame({"identifier" : identifier, 'response': response, 'sentiment': sentiment, "sync_ratio": sync_ratio, "interactions": interactions, "request": body, "identifier": identifier, "origin": origin})
        #    state_df.to_sql('interactions', con=connection, if_exists='append') 
        #    log.debug("Current state: {}".format(event_log))



        with open(DATA_FILENAME, mode='w', encoding='utf-8') as feedsjson:
            event_log.append(current_state)
            json.dump(event_log, feedsjson)
            
        return str(responce)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
    print(message.sid)
