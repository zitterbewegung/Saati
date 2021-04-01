#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .inference_functions import blenderbot400M, blenderbot3B, compute_sentiment

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


class Event(BaseModel):
    uuid: str = uuid.uuid4()
    aggregate_uuid = str
    timestamp: datetime = datetime.now()
    responses: List[str] = []
    sentiment: int = 1
    interactions: int = 1
    sync_ratio: float = 1
    state_machine: Any


class Saati(object):
    def __init__(self, name=uuid.uuid4(), debugMode=False):
        # No anonymous superheroes on my watch! Every narcoleptic superhero gets
        # a name. Any name at all. SleepyMan. SlumberGirl. You get the idea.
        self.name = name

        # How do we feel about the person.
        self.sentiment = 1

        # Interaction_number
        self.interactions = 1

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
        self.sync_ratio = self.sentiment / self.interactions

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


def answer_question(body, identifier, DATA_FILENAME_SUFFIX="state.json"):
    """
    >>> answer_question('hello')
    ' Hello! How are you doing today? I just got back from a walk with my dog.'
    """
    DATA_FILENAME = "{}_{}".format(identifier, DATA_FILENAME_SUFFIX)
    event_log = []
    log = logging.getLogger('saati.logic')
    log.debug('Response: {} Identifier {}, State file: {}'.format(body, identifier, DATA_FILENAME))
    log.info('restoring state')
    if os.path.exists(DATA_FILENAME):
        with open(DATA_FILENAME, mode='r') as feedsjson:
            event_log = json.load(feedsjson)
            #file_pi2 = open('state.json', 'r') 
            #state = file_pi2
    else:
        with open(DATA_FILENAME, mode='w', encoding='utf-8') as f:
            json.dump([], f)
    state = {}  
    if event_log != []:
        state = event_log[-1]
    sentiment = state.get('sentiment', 1)
    #sentiment = 1
    interactions = state.get('interactions', 1)

    #interactions = 1
    sync_ratio = sentiment / interactions
    responses = state.get('responses', [])

    instance_from_log = [
        str(responses),
        str(sentiment),
        str(sync_ratio),
        str(interactions),
        str(state.get('instance.state')),
    ]
    instance = Saati(uuid.uuid4(), instance_from_log)
    
    

    log.info("Computing reply")
    responce = blenderbot3B(body)[0]

    responses.append(responce)
    sentiment = sentiment + compute_sentiment(body)
    interactions = interactions + 1
    log.info(
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
        #talk("Hey, lets stay friends")
        instance.friendzone()
    #file = open('state.pkl', 'wb')
    # store the machine
    #dump = pickle.dumps(m)

    current_state = {'responses': responses,
                     'sentiment': sentiment,
                     'sync_ratio' : sync_ratio,
                     'interactions': interactions,
                     #'instance_path' : pickle.dumps(instance),
                     'request_time':  str(datetime.now())}
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

if __name__ == "__main__":
    import doctest
    doctest.testmod()
