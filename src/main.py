#!/usr/bin/env python3

from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional, Any
from fastapi import FastAPI
from fastapi import Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import csv, smtplib, uuid, logging, os, pickle, json
from logic import Saati, compute_sentiment, compute_sentiment

instance = Saati(uuid.uuid4())

class Event(BaseModel):
    uuid: str = uuid.uuid4()
    aggregate_uuid = str
    timestamp: datetime = datetime.now()
    responses: List[str] = []
    sentiment: int
    interactions: int
    sync_ratio: float
    state_machine: Any

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


#@app.get("/items/{item_id}")
#def read_item(item_id: int, q: Optional[str] = None):
#    return {"item_id": item_id, "q": q}


#@app.post("/events/")
#async def create_item(item: Item):
#    return item


import tempfile


@app.post('/process_utterance')
def pitch_track():
    import parselmouth

    # Save the file that was sent, and read it into a parselmouth.Sound
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(request.files['audio'].read())
        sound = parselmouth.Sound(tmp.name)

    # Calculate the pitch track with Parselmouth
    #pitch_track = sound.to_pitch().selected_array['frequency']

    # Convert the NumPy array into a list, then encode as JSON to send back
    return jsonify(list(pitch_track))


from logic import answer_question
import sys, logging, uuid

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

#define app routes
@app.get("/chatbot", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("chatbot.html", {"request": request})

#@app.route("/chatbot")
#def get():
#    return templates.TemplateResponse("index.html", {"request": request})
#    #return render_template("chatbot.html")

#class UserInput(BaseModel):
    

#function for the bot response
class Msg(BaseModel):
    msg: str

@app.get("/get")
def get_bot_response(msg: Msg):
    
    #resp = make_response()
    #if not request.cookies.get('userID'):
    #    resp.set_cookie('userID', session['identifier'])
    
    #cookieid = request.cookies.get('userID')
    user_identifier = "1" #session.get('identifier', uuid.uuid4())
    
    userText = msg #request.get('msg')

    inference = answer_question(userText, user_identifier, 'webchat')
    return Response(content=inference, media_type="text/plain")
    #return inference

    #return str(englishBot.get_response(userText))

#if __name__ == "__main__":
#    app.run(debug=True)
