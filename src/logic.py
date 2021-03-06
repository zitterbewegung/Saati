#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from inference_functions import blenderbot400M, blenderbot3B, compute_sentiment

import uuid, json, pickle, logging
from typing import List, Any

from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional
from transitions.extensions import GraphMachine as Machine
from transitions.extensions import HierarchicalMachine as Machine
from transitions import Machine
from pathlib import Path
import os, logging
import pandas as pd
from sqlalchemy import create_engine

def greetMe():
    CurrentHour = int(datetime.now().hour)
    if CurrentHour >= 0 and CurrentHour < 12:
        talk("Good Morning!")

    elif CurrentHour >= 12 and CurrentHour < 18:
        talk("Good Afternoon!")

    elif CurrentHour >= 18 and CurrentHour != 0:
        talk("Good Evening!")


def greet(name):
    return "Hello " + name + "!"


def journal_sleep(response: str):
    CurrentHour = int(datetime.now().hour)
    if CurrentHour >= 0 and CurrentHour < 9:
        talk(" How well did you sleep ? ")
    elif CurrentHour >= 10 and CurrentHour <= 12:
        talk(" Did you sleep in? ")
    return response

class Saati(object):
    def __init__(self, name=uuid.uuid4(), debugMode=False):
        # No anonymous superheroes on my watch! Every narcoleptic superhero gets
        # a name. Any name at all. SleepyMan. SlumberGirl. You get the idea.
        self.name = name

        # How do we feel about the person.
        self.sentiment = 1

        # Interaction_number
        self.interactions = 1
        self.positive_interactions = 1
        self.negative_interactions = 1


        self.sync_ratio = 1.0

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
        self.machine.add_ordered_transitions(conditions=["is_liked"])
        self.machine.add_transition(
            trigger="friendzone", source="*", conditions=["is_disliked"], dest=None
        )
        # Initialize models

    def update_sync_ratio(self):
        """ Dear Diary, today I saved Mr. Whiskers. Again. """
        self.sync_ratio = self.positive_interactions / self.negative_interactions

    @property
    def is_disliked(self):

        if 5 >= self.sync_ratio <= 11 and self.interactions > 10:
            return True
        else:
            return False

    @property
    def is_liked(self):
        # sync_ratio = self.sentiment / self.interaction_number
        return 5 < self.sync_ratio and self.sync_ratio < 15




def answer_question(incoming_msg, identifier, origin):
    """
    >>> answer_question('hello', '709c346d-4188-4a72-adeb-45308840c549', 'webchat')
    ' Hello! How are you doing today? I just got back from a walk with my dog.'
    """
    DATA_FILENAME = '{}-state.json'.format(identifier)
    
    event_log = []
    log = logging.getLogger('saati.logic')
    log.debug('Response: {} Identifier {}, State file: {}'.format(incoming_msg, identifier, DATA_FILENAME))
    log.info('restoring state')
    if os.path.exists(DATA_FILENAME):
        with open(DATA_FILENAME, mode='r', encoding='utf-8') as feedsjson:
            event_log = json.load(feedsjson)
            
    else:
        with open(DATA_FILENAME, mode='w', encoding='utf-8') as f:
            json.dump([], f)
    state = {}  
    if event_log != []:
        state = event_log[-1]

    state = {}
    if event_log != []:
        state = event_log[-1]
    sentiment = state.get("sentiment", 1)
    sync_ratio = state.get("sync_ratio" , 1)
    interactions = state.get("interactions", 1)
    positive_interactions = state.get("positive_interactions", 1)
    negative_interactions = state.get("negative_interactions", 1)
    # interactions = 1


    level_counter = state.get("level_counter", 1)
    responses = state.get("responses", [])

    instance_from_log = [
        str(responses),
        str(sentiment),
        str(sync_ratio),
        str(positive_interactions),
        str(negative_interactions)
        #str(state.get('instance.state')),
    ]
    instance = Saati(uuid.uuid4(), instance_from_log)

    log.info("Computing reply")
    responce = blenderbot3B(incoming_msg)[0] 

    responses.append(responce)
    sentiment = compute_sentiment(incoming_msg)

    if sentiment == 'POSITIVE':
        positive_interactions = positive_interactions + 1

    if sentiment == 'NEGATIVE':
        negative_interactions = negative_interactions + 1
        
        
    sync_ratio = positive_interactions / negative_interactions

    interactions = interactions + 1
    logging.info(
        "Incoming Message: {} Responses: {} Sentiment: {}  Sync ratio: {} Interactions: {} Positive Interactions {} Negative Interactions {} level_counter {} Current State {}, response_sentiment {} timestamp {}".format(
            incoming_msg,
            responses,
            sentiment,
            sync_ratio,
            interactions,
            positive_interactions,
            negative_interactions,
            level_counter,
            instance.state,
            compute_sentiment(responce),
            str(datetime.now()),
            identifier,
            origin,
        )
    )
    current_state = {
        "responses": responses,
        "sentiment": sentiment,
        "sync_ratio": sync_ratio,
        "incoming_msg" : incoming_msg,
        "interactions": interactions,

        "positive_interactions": positive_interactions,
        "negative_interactions": positive_interactions,

        "level_counter" : level_counter,
        "response_sentiment": compute_sentiment(responce),
        "timestamp": str(datetime.now()),
        "identifier": identifier,
        "origin": origin,
    }

    if sync_ratio > 5 and sync_ratio < 11: 
        level_counter = level_counter + 1  
        instance.next_state()
    if sync_ratio > 11 or sync_ratio < 5:
        level_counter = level_counter - 1   

    with open(DATA_FILENAME, mode='w', encoding='utf-8') as feedsjson:
        event_log.append(current_state)
        json.dump(event_log, feedsjson)

        
    return responce


class CoffeeLevel(object):

    states = [
        "standing",
        "walking",
        {"name": "caffeinated", "children": ["dithering", "running"]},
    ]
    transitions = [
        ["walk", "standing", "walking"],
        ["stop", "walking", "standing"],
        ["drink", "*", "caffeinated"],
        ["walk", ["caffeinated", "caffeinated_dithering"], "caffeinated_running"],
        ["relax", "caffeinated", "standing"],
    ]

    def __init__(self, name=uuid.uuid4(), debugMode=False):

        machine = Machine(
            states=states,
            transitions=transitions,
            initial="standing",
            ignore_invalid_triggers=True,
        )

#if __name__ == "__main__":
#    answer_question("hello", "709", "temp", "test_state.json")
#    import doctest
#    doctest.testmod()
