#!/usr/bin/env python3
import torch
import numpy as np
from scipy.io.wavfile import write
from transformers import (
    TFAutoModelWithLMHead,
    AutoTokenizer,
    pipeline,
    BlenderbotTokenizer,
    BlenderbotForConditionalGeneration,
    Conversation,
)
from transformers import BlenderbotSmallTokenizer, BlenderbotForConditionalGeneration
from transformers import pipeline
import uuid, json, pickle
from typing import List, Any

from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional


#!pip install streamlit
#!pip install transitions[diagrams]
#!pip install graphviz pygraphviz
#!brew install graphviz
# from transitions.extensions import GraphMachine as Machine
from transitions import Machine

import random
from datetime import datetime

# Set up logging; The basic log level will be DEBUG
import logging
import speech_recognition as sr
import torch
import numpy as np

import streamlit as st

logging.basicConfig(level=logging.INFO)

# engine = pyttsx3.init("nsss")


class Saati(object):
    def __init__(self, name, debugMode=False):
        # No anonymous superheroes on my watch! Every narcoleptic superhero gets
        # a name. Any name at all. SleepyMan. SlumberGirl. You get the idea.
        self.name = name

        # How do we feel about the person.
        self.sentiment = 1

        # Interaction_number
        self.interaction_number = 0

        # Figure out outcome that would put you in the friendzone?
        # self.love_vector = self.impression_points * random.randrange(20) / self.interaction_number

        # Initialize the state machine

        # states represent where you are.
        states = [
            "initializemodels",
            "meetup",
            "hangingout",
            "sleeping",
            "wake_up",
            "leave",
        ]
        self.machine = Machine(model=self, states=states, initial="initializemodels")
        self.machine.add_ordered_transitions()
        self.machine.add_transition(trigger="friendzone", source="*", dest=None)
        # Initialize models


def GivenCommand(test_mode=True):
    Input = ""
    if test_mode:
        Input = input("Resp>>")
        return Input
    else:
        k = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            k.pause_threshold = 1
            audio = k.listen(source)
        try:
            Input = k.recognize_google(audio, language="en-us")
            talk("You: " + Input + "\n")
        except sr.UnknownValueError:
            talk("Gomen! I didn't get that! Try typing it here!")
            Input = str(input("Command: "))
    return Input


def smalltalk(utterance: str) -> List[str]:

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logging.info("starting smalltalk")
    mname = "facebook/blenderbot-3B"
    model = BlenderbotForConditionalGeneration.from_pretrained(mname)
    model.to(device)
    tokenizer = BlenderbotTokenizer.from_pretrained(mname)
    inputs = tokenizer([utterance], return_tensors="pt").to(device)
    reply_ids = model.generate(**inputs)
    responses = [
        tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for g in reply_ids
    ]
    return responses


def is_a_question(utterance: str) -> bool:
    START_WORDS = [
        "who",
        "what",
        "when",
        "where",
        "why",
        "how",
        "is",
        "can",
        "does",
        "do",
    ]
    for word in START_WORDS:
        if word in START_WORDS:
            return True
    return false


def talk(text: str):
    logging.info("starting waveglow")
    device_to_use = "cuda" if torch.cuda.is_available() else "cpu"
    if device_to_use:
        logging.info("Saati: " + text)

        engine.say(text)
        engine.runAndWait()
    else:

        waveglow = torch.hub.load(
            "nvidia/DeepLearningExamples:torchhub", "nvidia_waveglow"
        )
        waveglow = waveglow.remove_weightnorm(waveglow)
        waveglow = waveglow.to(device_to_use)
        waveglow.eval()
        tacotron2 = torch.hub.load(
            "nvidia/DeepLearningExamples:torchhub", "nvidia_tacotron2"
        )
        tacotron2 = tacotron2.to(device_to_use)
        tacotron2.eval()
        # preprocessing
        sequence = np.array(tacotron2.text_to_sequence(text, ["english_cleaners"]))[
            None, :
        ]
        sequence = torch.from_numpy(sequence).to(
            device=device_to_use, dtype=torch.int64
        )

        # run the models
        with torch.no_grad():
            _, mel, _, _ = tacotron2.infer(sequence)
            audio = waveglow.infer(mel)
            audio_numpy = audio[0].data.cpu().numpy()
            rate = 22050

            write("/tmp/audio.wav", rate, audio_numpy)
            with open("/tmp/audio.wav", "rb") as f:
                b = f.read()
                play_obj = sa.play_buffer(b, 2, 2, 22050)

                play_obj.wait_done()

    # return audio


def GivenCommand(test_mode=False):
    Input = ""
    if test_mode:
        Input = input("Resp>>")
    else:
        k = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            k.pause_threshold = 1
            audio = k.listen(source)
        try:
            Input = k.recognize_google(audio, language="en-us")
            print("You: " + Input + "\n")
        except sr.UnknownValueError:
            talk("Gomen! I didn't get that! Try typing it here!")
            Input = str(input("Command: "))
    return Input


class Event(BaseModel):
    uuid: str = uuid.uuid4()
    timestamp: datetime = datetime.now()
    responses: List[str] = []
    sentiment: int = 1
    interactions: int = 1
    sync_ratio: float = 1
    state_machine: Any


# function to add to JSON
def write_json(data, filename="event_log.json"):
    with open(filename, "a+") as f:
        json.dump(data, f, indent=4)
        f.write("\n")


def compute_sentiment(utterance: str) -> float:
    nlp = pipeline("sentiment-analysis")
    result = nlp(utterance)
    score = result[0]["score"]
    if result[0]["label"] == "NEGATIVE":
        score = score * -1

    logging.info("The score was {}".format(score))
    return score


def local_ingest():
    """
    If pos or neg pos 5 to 1 relationship doesn't continue
    If exceeds 11 pos 1 neg no challenge
    you wlant not bliss but
    """

    instance = Saati(uuid.uuid4())

    user_input = GivenCommand()

    from pathlib import Path
    import pickle

    my_file = Path("event_log.json")
    state_machine = pickle.dumps(instance)
    current_state = Event()

    if my_file.is_file():
        with open("event_log.json", "r") as f:
            data = f.read()
            save_state = json.loads(data)
    write_json(current_state.json())
    # if my_file.is_file():
    #    current_state = pickle.loads(my_file)
    # else:
    #    current_state = pickle.dumps(current_state)

    while True:
        # instance.get_graph().draw('my_state_diagram.png', prog='dot')

        logging.info("Computing reply")
        responce = smalltalk(user_input)[0]
        talk(responce)
        current_state.responses.append(responce)
        current_state.sentiment = current_state.sentiment + compute_sentiment(
            user_input
        )
        current_state.interactions = current_state.interactions + 1
        current_state.sync_ratio = current_state.sentiment / current_state.interactions
        logging.info(
            "Responses: {} Sentiment: {}  Sync ratio: {} Interactions: {}	| Current State {}".format(
                str(current_state.responses),
                str(current_state.sentiment),
                str(current_state.sync_ratio),
                str(current_state.interactions),
                str(current_state.state_machine),
            )
        )

        if 5 <= current_state.sync_ratio <= 11:
            instance.next_state()
        else:
            print("Hey, lets stay friends")
            instance.friendzone()
            return
    # current_state.state_machine = pickle.dumps(instance)

def answer_question(body):
    instance = Saati(uuid.uuid4())

    sentiment = 1

    interactions = 1
    sync_ratio = sentiment / interactions

    logging.info("Computing reply")
    responce = ["Hello"]  # smallertalk(body)  # [0]
    # resp = MessagingResponse()
    current_state = Event(
        input=body,
        output=responce,
        sentiment=sentiment,
        sync_ratio=sync_ratio,
        interactions=interactions,
        state_machine=instance,
    )

    from pathlib import Path

    my_file = Path("event_log.dat")
    if my_file.is_file():
        save_state = pickle.load(open("event_log.dat", "rb"))
        pickled_state_machine = save_state.get("state_machine")
        state_machine = pickle.loads(pickled_state_machine)
        interactions = current_state.interactions
        print(interactions)

    sentiment = sentiment + compute_sentiment(body)
    interactions = interactions + 1

    logging.info(
        "Responses: {} Sentiment: {}  Sync ratio: {} Interactions: {}	| Current State {}".format(
            str(responce),
            str(sentiment),
            str(sync_ratio),
            str(interactions),
            str(instance.state),
        )
    )
    dump = pickle.dumps(instance)

    save_state = {"state_machine": dump, "current_state": current_state.dict()}

    with open("event_log.dat", "wb") as file:
        data = pickle.dumps(save_state)
        file.write(data)

    # with open("save_state.json", "r+") as file:
    # 	 data = json.load(file)
    # 	 data.update(save_state)
    # 	 file.seek(0)
    # 	 json.dump(data, file)

    # my_dict = {'1': 'aaa', '2': 'bbb', '3': 'ccc'}

    if 5 >= sync_ratio <= 11 or interactions < 10:
        instance.next_state()
    else:
        instance.friendzone()

    return responce


if __name__ == "__main__":
    local_ingest()
    # st.title('saati Demo')
    # starting_text = st.text_area('Hello!')

    # if starting_text:
    # 	response = smalltalk(starting_text)
    # 	st.markdown(f'Saati: {response}')
    # data = "My data read from the Web"
    # print(data)
    # modified_data = process_data(data)
    # print(modified_data)
