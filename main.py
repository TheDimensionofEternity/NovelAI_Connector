"""
Title: NovelAI Connector
author: TheDimensionOfEternity
author_url: https://github.com/TheDimensionofEternity
licence: The Unlicense

Make sure that the novelai-api module is installed in your environment
before installing this function. it won't work otherwise.

In modelfiles, you can specify the Assistant's name with the following
syntax on the first line of the system prompt.

Assistant=My Custom Model

This is my first time publically submitting code and I haven't rigorously
tested it, there are likely to be bugs.
"""

from pydantic import BaseModel, Field
from icecream import ic
import asyncio
from novelai_api import NovelAIAPI
from fastapi import Request

## START OF BOILERPLATE

import asyncio
import json
from datetime import datetime
from logging import Logger, StreamHandler
from os import environ as env
from pathlib import Path
from typing import Any, Optional, List, Dict, Callable, Union, Iterable, AsyncIterable

from aiohttp import ClientSession
from msgpackr.constants import UNDEFINED

from novelai_api import NovelAIAPI, Keystore
from novelai_api.utils import get_encryption_key
from novelai_api.Msgpackr_Extensions import (
    Ext20, 
    Ext30, 
    Ext31, 
    Ext40, 
    Ext41, 
    Ext42,
)
import novelai_api.utils
from icecream import ic
import atexit

class API:
    """
    Boilerplate for the redundant parts.
    Using the object as a context manager will automatically login using the environment variables
    ``NAI_USERNAME`` and ``NAI_PASSWORD``.

    Usage:

    .. code-block:: python

        async with API() as api:
            api = api.api
            encryption_key = api.encryption_key
            logger = api.logger
            ...  # Do stuff


    A custom base address can be passed to the constructor to replace the default
    (:attr:`BASE_ADDRESS <novelai_api.NovelAI_API.NovelAIAPI.BASE_ADDRESS>`)
    """

    _username: str
    _password: str
    _token: str 
    _session: ClientSession

    logger: Logger
    api: Optional[NovelAIAPI]

    def __init__(
        self,
        base_address: Optional[str] = None,
        username:str = "",
        password:str = "",
        token:str = "",
    ):
        dotenv = Path(".env")
        if dotenv.exists():
            with dotenv.open("r") as f:
                for line in f:
                    if "=" in line:
                        key, value = line.strip().split("=", 1)
                        env[key] = value.strip()
        
        if ((username != "" or "NAI_USERNAME" in env) or
            (password != "" or "NAI_PASSWORD" in env)):
            if not (username != "" or "NAI_USERNAME" in env):
                raise RuntimeError(
                    "Please ensure that NAI_USERNAME is set in your environment"
                )

            if not (password != "" or "NAI_PASSWORD" in env):
                raise RuntimeError(
                    "Please ensure that NAI_PASSWORD is set in your environment"
                )

            self._username = username if username != "" else env["NAI_USERNAME"]
            self._password = password if password != "" else env["NAI_PASSWORD"]
            self._using_token = False
        else:
            if not (token != "" or "NAI_TOKEN" in env):
                raise RuntimeError(
                    "Please ensure that NAI_TOKEN is set in your environment"
                )
            self._using_token = True
            self._token = token if isinstance(token, str) else env["NAI_TOKEN"]

        self.logger = Logger("NovelAI")
        self.logger.addHandler(StreamHandler())

        self.api:NovelAIAPI = NovelAIAPI(logger=self.logger)
        if base_address is not None:
            self.api.BASE_ADDRESS = base_address

        atexit.register(self.on_exit)

    @property
    def encryption_key(self) -> bytes:
        if self._using_token:
            return bytes()
        return get_encryption_key(self._username, self._password)

    async def __aenter__(self):
        self._session = ClientSession()
        await self._session.__aenter__()

        self.api.attach_session(self._session)
        if self._using_token:
            await self.api.high_level.login_with_token(self._token)
        else:
            await self.api.high_level.login(self._username, self._password)

        return self
    
    def on_exit(self):
        asyncio.run(self.__aexit__())

    async def __aexit__(self, exc_type=None, exc_val=None, exc_tb=None):
        await self._session.__aexit__(exc_type, exc_val, exc_tb)

class JSONEncoder(json.JSONEncoder):
    """
    Extended JSON encoder to support bytes
    """

    def default(self, o: Any) -> Any:
        if isinstance(o, bytes):
            return o.hex()
        if o is UNDEFINED:
            return f"<UNDEFINED> {repr(o)}"
        if isinstance(o, datetime):
            return o.isoformat()

        return super().default(o)

def dumps(e: Any) -> str:
    """
    Shortcut to a configuration of json.dumps for consistency
    """

    return json.dumps(e, indent=4, ensure_ascii=False, cls=JSONEncoder)

def document_wrapper(func:Callable):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TypeError:
            del novelai_api.utils.unpacker
            novelai_api.utils.unpacker = novelai_api.utils.Unpacker()
            novelai_api.utils.unpacker.register_extensions(
                Ext20, Ext30, Ext31, Ext40, Ext41, Ext42
            )
            return func(*args, **kwargs)
    return wrapper

@document_wrapper
def decrypt_user_data(
        items: List[Dict[str, Any]] | Dict[str, Any], 
        keystore: Keystore, 
        uncompress_document: bool = False
) -> None:

    return novelai_api.utils.decrypt_user_data(
        items=items,
        keystore=keystore,
        uncompress_document=uncompress_document
    )

### END OF BOILERPLATE

from novelai_api.BanList import BanList
from novelai_api.BiasGroup import BiasGroup
from novelai_api.GlobalSettings import GlobalSettings
from novelai_api.Preset import PREAMBLE, Model, Preset, Order
from novelai_api.Tokenizer import Tokenizer
from novelai_api.utils import b64_to_tokens

class Pipe:
    class Valves(BaseModel):
        # NOVELAI_EMAIL: str = Field(
        #     default="",
        #     description="Email of account used to authenticate authenticating "
        #     "requests to the NovelAI API.",
        # )
        # NOVELAI_PASSWORD: str = Field(
        #     default="",
        #     description="Password of account used to authenticate authenticating "
        #     "requests to the NovelAI API.",
        # )
        NOVELAI_API_KEY: str = Field(
            default="",
            description="API key for authenticating requests to the NovelAI API.",
        )
        USER_FORMATTING: str = Field(
            default="\n{user}: {data}\n",
            description="",
        )
        ASSISTANT_FORMATTING: str = Field(
            default="\n{assistant}: {data}\n",
            description="",
        )
        SYSTEM_FORMATTING: str = Field(
            default="\n {data}\n",
            description="",
        )

    def __init__(self):
        self.valves = self.Valves()

    def format_message(self, message:list, user:dict, assistant:str="Assistant") -> str:
        formatted_message:str = ""
        for item in message + [{"role":"assistant","content":""}]:
            content=item["content"]
            match item["role"]:
                case "user":
                    formatted_message += self.valves.USER_FORMATTING.format(
                        user=user["name"].title(),
                        assistant=assistant.title(),
                        data=content
                    )
                case "assistant":
                    formatted_message += self.valves.ASSISTANT_FORMATTING.format(
                        user=user["name"].title(),
                        assistant=assistant.title(),
                        data=content
                    )
                case "system":
                    formatted_message += self.valves.SYSTEM_FORMATTING.format(
                        user=user["name"].title(),
                        assistant=assistant.title(),
                        data=content.format(
                            user=user["name"].title(),
                            assistant=assistant,
                        )
                    )
                case _:
                    raise ValueError("Uhh, how'd this happen? invalid role.")
        return formatted_message.strip()
    
    def pipes(self):
        # if (not self.valves.NOVELAI_EMAIL and not self.valves.NOVELAI_PASSWORD
        if (not self.valves.NOVELAI_API_KEY):
            return [
                {
                    "id": "error",
                    "name": "Proper NovelAI Authentication not provided."
                },
            ]

        erato_models = [
            {
                "id": f"Erato/{preset_name.replace(' ','-')}",
                "name": f"{preset_name} (Erato)",
            }
            for preset_name in self.ERATO_PRESETS
        ]
        kayra_models = [
            {
                "id": f"Kayra/{preset_name.replace(' ','-')}",
                "name": f"{preset_name} (Kayra)",
            }
            for preset_name in self.ERATO_PRESETS
        ]
        clio_models = [
            {
                "id": f"Clio/{preset_name.replace(' ','-')}",
                "name": f"{preset_name} (Clio)",
            }
            for preset_name in self.CLIO_PRESETS
        ]
        models = erato_models + kayra_models + clio_models
        return models[:]

    @staticmethod
    async def generate_stream(
        api: NovelAIAPI,
        prompt: Union[List[int], str],
        model: Model,
        preset: Preset,
        global_settings: GlobalSettings,
        bad_words: Optional[Union[Iterable[BanList], BanList]] = None,
        biases: Optional[Union[Iterable[BiasGroup], BiasGroup]] = None,
        prefix: Optional[str] = None,
        stop_sequences: Optional[Union[List[int], str]] = None,
        **kwargs,
    ) -> AsyncIterable[Dict[str, Any]]:

        bytes_per_token:int = 2
        if model == Model.Erato:
            bytes_per_token = 4

        prev_tokens = []

        async for e in api.high_level._generate(
            prompt,
            model,
            preset,
            global_settings,
            bad_words,
            biases=biases,
            prefix=prefix,
            stop_sequences=stop_sequences,
            stream=True,
            **kwargs
        ):
            def isMatch(token) -> int:
                nonlocal prev_tokens
                is_match:int = 0
                for sequence in stop_sequences:
                    sequence_additional = [sequence[0]]+[34184]+sequence[1:]
        
                    if prev_tokens == []:
                        break
                    elif (prev_tokens == sequence):
                        is_match = 2
                    elif (prev_tokens == sequence[:len(prev_tokens)]):
                        is_match = 1
                        break
                    elif (prev_tokens == sequence_additional[:len(prev_tokens)]):
                        del prev_tokens[1]
                        is_match = 1
                        break
                
                return is_match

            def do_work(token):
                nonlocal prev_tokens
                output_data = ""
                prev_tokens.extend(token)
                is_match = isMatch(token)
                if not is_match:
                    output_data += Tokenizer.decode(
                        model,
                        prev_tokens
                    )
                    prev_tokens = []
                    prev_tokens.extend(token)
                    if not isMatch(token):
                        prev_tokens=[]
                elif is_match == 2:
                    output_data += Tokenizer.decode(
                        model, 
                        [prev_tokens[0]]
                    ).removesuffix("\n")


                return output_data

            token = b64_to_tokens(json.loads(e)["token"],bytes_per_token)
            yield do_work(token)
                
    
    @classmethod
    def generate_stop_sequences(cls, model:Model, *roles:str) -> List[List[int]]:
        stop_sequences:List[str] = []
        for role in roles:
            for sequence in cls.STOP_SEQUENCES:
                stop_sequences.append(sequence.format(role=role.title()))
        
        stop_sequences += cls.ADDITIONAL_STOP_SEQUENCES

        tokenized_stop_sequences:List[List[int]] = []
        for sequence in stop_sequences:
            tokens = Tokenizer.encode(
                model=model,
                o=sequence
            )
            if model == Model.Erato:
                tokens = tokens[1:]
            tokenized_stop_sequences.append(tokens)
        return tokenized_stop_sequences

    async def pipe(self, body:dict, __user__:dict):
        MESSAGES:List[Dict[str:str]] = body["messages"]
        model:str = body["model"].removeprefix("novelai.")
        STREAM:bool = body["stream"]
        assistant = "Assistant"
        if str(MESSAGES[0]["content"]).split("\n",1)[0].startswith("Assistant="):
            assistant = MESSAGES[0]["content"].split("\n",1)[0].split("=",1)[-1]
            MESSAGES[0]["content"] = MESSAGES[0]["content"].split("\n",1)[1]

        async with API(
            username="",
            password="",
            token=self.valves.NOVELAI_API_KEY,
        ) as api_handler:
            api = api_handler.api
            logger = api_handler.logger

            model_name:str = model.split("/")[0]
            model_preset:str = model.split("/")[1]

            match model_name:
                case "Erato": 
                    preset_gen:dict = self.PRESETS["Erato-"+model_preset.replace(" ","-")]
                    model = Model.Erato
                case "Kayra": 
                    preset_gen:dict = self.PRESETS[model_preset.replace(" ","-")+"-Kayra"]
                    model = Model.Kayra
                case "Clio" : 
                    preset_gen:dict = self.PRESETS["Clio-"+model_preset.replace(" ","-")]
                    model = Model.Clio
                case _:
                    yield "error"

            preset_gen["order"] = [self.INT_TO_ORDER[o] for o in preset_gen["order"]]

            prompt = PREAMBLE[model] + self.format_message(MESSAGES,__user__,assistant)
            preset = Preset(
                model_preset,
                model,
            )

            for key, value in preset_gen.items():
                if hasattr(preset, key):
                    setattr(preset, key, value)
                    if key == 'order':
                        preset.sampling_options = [True]*len(value)


            global_settings = GlobalSettings(num_logprobs=GlobalSettings.NO_LOGPROBS)
            global_settings.bias_dinkus_asterism = True
            global_settings.rep_pen_whitelist = True
            global_settings.generate_until_sentence = True

            bad_words: Optional[BanList] = None
            if bad_words is not None:
                bad_words.add(
                    "<|startoftext|>", 
                    "<|endoftext|>",
                    "<||ENDOFFTEXT||>",
                    "<|endoftext|>",
                )

            bias_groups: List[BiasGroup] = []

            module=None
            stop_sequence=self.generate_stop_sequences(
                model,
                "***",
                __user__["name"]+":",
                f"{assistant}:",
            )
        


            bytes_per_token = 2
            if model == Model.Erato:
                bytes_per_token = 4

            if not STREAM:
                gen = await api.high_level.generate(
                    prompt,
                    model,
                    preset,
                    global_settings,
                    bad_words=bad_words,
                    biases=bias_groups,
                    prefix=module,
                    stop_sequences=stop_sequence,
                )

                yield Tokenizer.decode(model, b64_to_tokens(gen["output"],bytes_per_token))
            else:

                async for e in self.generate_stream(
                    api,
                    prompt,
                    model,
                    preset,
                    global_settings,
                    bad_words,
                    biases=bias_groups,
                    prefix=module,
                    stop_sequences=stop_sequence,
                ):
                    yield e



    INT_TO_ORDER = {
        0:  Order.Temperature,
        1:  Order.Top_K,
        2:  Order.Top_P,
        3:  Order.TFS,
        4:  Order.Top_A,
        5:  Order.Typical_P,
        8:  Order.Mirostat,
        9:  Order.Unified,
        10: Order.Min_p,
    }

    ERATO_PRESETS = [
        "Shosetsu",
        "Wilder",
        "Dragonfruit",
        "Zany Scribe",
        "Golden Arrow",
    ]
    KAYRA_PRESETS = [
        "Fresh Coffee",
        "Asper",
        "Writers Daemon",
        "Carefree",
        "Stelenes",
    ]
    """OLD_KAYRA_PRESETS = [
        "Tesseract",
        "Blended Coffee",
        "CosmicCube",
        "Tea Time",
        "Pilotfish",
        "Green Active Writer",
        "Blook",
        "Pro Writer",
    ]"""
    CLIO_PRESETS = [
        "Vingt-Un",
        "Long Press",
        "Edgewise",
        "Fresh Coffee",
        "Talker Chat",
    ]
    """OLD_CLIO_PRESETS = [
        "Keelback",
    ]"""

    STOP_SEQUENCES = [
        '\n{role}',
        '.\n{role}',
        '!\n{role}',
        '?\n{role}',
        '*\n{role}',
        '\"\n{role}',
        '_\n{role}',
        '...\n{role}',
        '.\"\n{role}',
        '?\"\n{role}',
        '!\"\n{role}',
        '.*\n{role}',
        ')\n{role}',
        '.)\n{role}',
    ]
    ADDITIONAL_STOP_SEQUENCES = [
        "\n***"
    ]

    PRESETS = {
        "Fresh-Coffee-Kayra": {
            "order": [
                0,
                1,
                2,
                3
            ],
            "temperature": 1,
            "max_length": 150,
            "min_length": 1,
            "top_k": 25,
            "top_p": 1,
            "tail_free_sampling": 0.925,
            "repetition_penalty": 1.9,
            "repetition_penalty_range": 768,
            "repetition_penalty_slope": 1,
            "repetition_penalty_frequency": 0.0025,
            "repetition_penalty_presence": 0.001,
            "use_cache": False,
            "return_full_text": False,
            "prefix": "vanilla",
            "phrase_rep_pen": "off",
            "max_context": 7800,
            "min_p": 0,
            "math1_temp": 1,
            "math1_quad": 0,
            "math1_quad_entropy_scale": 0
        },
        "Asper-Kayra": {
            "order": [
                5,
                0,
                1,
                3
            ],
            "temperature": 1.16,
            "max_length": 150,
            "min_length": 1,
            "top_k": 175,
            "typical_p": 0.96,
            "tail_free_sampling": 0.994,
            "repetition_penalty": 1.68,
            "repetition_penalty_range": 2240,
            "repetition_penalty_slope": 1.5,
            "repetition_penalty_frequency": 0,
            "repetition_penalty_presence": 0.005,
            "use_cache": False,
            "return_full_text": False,
            "prefix": "vanilla",
            "phrase_rep_pen": "medium",
            "max_context": 7800,
            "min_p": 0,
            "math1_temp": 1,
            "math1_quad": 0,
            "math1_quad_entropy_scale": 0
        },
        "Erato-Wilder": {
            "max_context": 8000,
            "temperature": 1,
            "max_length": 150,
            "min_length": 1,
            "top_k": 300,
            "top_p": 0.98,
            "top_a": 0.004,
            "typical_p": 0.96,
            "tail_free_sampling": 0.96,
            "repetition_penalty": 1.48,
            "repetition_penalty_range": 2240,
            "repetition_penalty_slope": 0.64,
            "repetition_penalty_frequency": 0,
            "repetition_penalty_presence": 0,
            "phrase_rep_pen": "medium",
            "mirostat_lr": 1,
            "mirostat_tau": 0,
            "math1_temp": -0.0485,
            "math1_quad": 0.145,
            "math1_quad_entropy_scale": 0,
            "min_p": 0.02,
            "order": [
                9,
                10
            ]
        },
        "Erato-Dragonfruit": {
            "max_context": 8000,
            "temperature": 1.37,
            "max_length": 150,
            "min_length": 1,
            "top_k": 0,
            "top_p": 1,
            "top_a": 0.1,
            "typical_p": 0.875,
            "tail_free_sampling": 0.87,
            "repetition_penalty": 3.25,
            "repetition_penalty_range": 6000,
            "repetition_penalty_slope": 3.25,
            "repetition_penalty_frequency": 0,
            "repetition_penalty_presence": 0,
            "phrase_rep_pen": "off",
            "mirostat_lr": 0.2,
            "mirostat_tau": 4,
            "math1_temp": 0.9,
            "math1_quad": 0.07,
            "math1_quad_entropy_scale": -0.05,
            "min_p": 0.035,
            "order": [
                0,
                5,
                9,
                10,
                8,
                4
            ]
        },
        "Edgewise-Clio": {
            "order": [
                4,
                0,
                5,
                3,
                2
            ],
            "temperature": 1.09,
            "max_length": 150,
            "min_length": 1,
            "top_p": 0.969,
            "top_a": 0.09,
            "typical_p": 0.99,
            "tail_free_sampling": 0.969,
            "repetition_penalty": 1.09,
            "repetition_penalty_range": 8192,
            "repetition_penalty_slope": 0.069,
            "repetition_penalty_frequency": 0.006,
            "repetition_penalty_presence": 0.009,
            "use_cache": False,
            "return_full_text": False,
            "prefix": "vanilla",
            "phrase_rep_pen": "very_light",
            "max_context": 7800,
            "min_p": 0,
            "math1_temp": 1,
            "math1_quad": 0,
            "math1_quad_entropy_scale": 0
        },
        "Vingt-Un-Clio": {
            "order": [
                0,
                5,
                3,
                2,
                1
            ],
            "temperature": 1.21,
            "max_length": 40,
            "min_length": 1,
            "top_k": 0,
            "top_p": 0.912,
            "top_a": 1,
            "typical_p": 0.912,
            "tail_free_sampling": 0.921,
            "repetition_penalty": 1.21,
            "repetition_penalty_range": 321,
            "repetition_penalty_slope": 3.33,
            "repetition_penalty_frequency": 0.00621,
            "repetition_penalty_presence": 0,
            "use_cache": False,
            "return_full_text": False,
            "prefix": "vanilla",
            "phrase_rep_pen": "very_light",
            "max_context": 7800,
            "min_p": 0,
            "math1_temp": 1,
            "math1_quad": 0,
            "math1_quad_entropy_scale": 0
        },
        "Fresh-Coffee-Clio": {
            "order": [
                0,
                1,
                2,
                3
            ],
            "temperature": 1,
            "max_length": 40,
            "min_length": 1,
            "top_k": 25,
            "top_p": 1,
            "top_a": 0,
            "typical_p": 1,
            "tail_free_sampling": 0.925,
            "repetition_penalty": 1.9,
            "repetition_penalty_range": 768,
            "repetition_penalty_slope": 3.33,
            "repetition_penalty_frequency": 0.0025,
            "repetition_penalty_presence": 0.001,
            "use_cache": False,
            "return_full_text": False,
            "prefix": "vanilla",
            "phrase_rep_pen": "very_light",
            "max_context": 7800,
            "min_p": 0,
            "math1_temp": 1,
            "math1_quad": 0,
            "math1_quad_entropy_scale": 0
        },
        "Writers-Daemon-Kayra": {
            "order": [
                8,
                0,
                5,
                3,
                2,
                4
            ],
            "temperature": 1.5,
            "max_length": 150,
            "min_length": 1,
            "top_a": 0.02,
            "top_p": 0.95,
            "typical_p": 0.95,
            "tail_free_sampling": 0.95,
            "mirostat_lr": 0.25,
            "mirostat_tau": 5,
            "repetition_penalty": 1.625,
            "repetition_penalty_range": 2016,
            "repetition_penalty_frequency": 0,
            "repetition_penalty_presence": 0,
            "use_cache": False,
            "return_full_text": False,
            "prefix": "vanilla",
            "phrase_rep_pen": "very_aggressive",
            "max_context": 7800,
            "min_p": 0,
            "math1_temp": 1,
            "math1_quad": 0,
            "math1_quad_entropy_scale": 0
        },
        "Talker-Chat-Clio": {
            "order": [
                1,
                5,
                0,
                2,
                3,
                4
            ],
            "temperature": 1.5,
            "max_length": 150,
            "min_length": 1,
            "top_k": 10,
            "top_p": 0.75,
            "top_a": 0.08,
            "typical_p": 0.975,
            "tail_free_sampling": 0.967,
            "repetition_penalty": 2.25,
            "repetition_penalty_range": 8192,
            "repetition_penalty_slope": 0.09,
            "repetition_penalty_frequency": 0,
            "repetition_penalty_presence": 0.005,
            "use_cache": False,
            "return_full_text": False,
            "prefix": "vanilla",
            "phrase_rep_pen": "very_light",
            "max_context": 7800,
            "min_p": 0,
            "math1_temp": 1,
            "math1_quad": 0,
            "math1_quad_entropy_scale": 0
        },
        "Tesseract-Kayra": {
            "order": [
                0,
                5
            ],
            "temperature": 0.895,
            "max_length": 150,
            "min_length": 1,
            "typical_p": 0.9,
            "repetition_penalty": 2,
            "repetition_penalty_slope": 3.2,
            "repetition_penalty_frequency": 0,
            "repetition_penalty_presence": 0,
            "repetition_penalty_range": 4048,
            "use_cache": False,
            "return_full_text": False,
            "prefix": "vanilla",
            "phrase_rep_pen": "aggressive",
            "max_context": 7800,
            "min_p": 0,
            "math1_temp": 1,
            "math1_quad": 0,
            "math1_quad_entropy_scale": 0
        },
        "Long-Press-Clio": {
            "order": [
                0,
                4,
                1,
                5,
                3
            ],
            "temperature": 1.155,
            "max_length": 40,
            "min_length": 1,
            "top_k": 25,
            "top_a": 0.3,
            "top_p": 1,
            "typical_p": 0.96,
            "tail_free_sampling": 0.895,
            "repetition_penalty": 1.0125,
            "repetition_penalty_range": 2048,
            "repetition_penalty_slope": 3.33,
            "repetition_penalty_frequency": 0.011,
            "repetition_penalty_presence": 0.005,
            "use_cache": False,
            "return_full_text": False,
            "prefix": "vanilla",
            "phrase_rep_pen": "very_light",
            "max_context": 7800,
            "min_p": 0,
            "math1_temp": 1,
            "math1_quad": 0,
            "math1_quad_entropy_scale": 0
        },
        "Keelback-Clio": {
            "order": [
                4,
                5,
                0,
                3
            ],
            "temperature": 1.18,
            "max_length": 40,
            "min_length": 1,
            "top_a": 0.022,
            "top_k": 0,
            "top_p": 1,
            "typical_p": 0.9,
            "tail_free_sampling": 0.956,
            "repetition_penalty": 1.25,
            "repetition_penalty_range": 4096,
            "repetition_penalty_slope": 0.9,
            "repetition_penalty_frequency": 0,
            "repetition_penalty_presence": 0,
            "use_cache": False,
            "return_full_text": False,
            "prefix": "vanilla",
            "phrase_rep_pen": "very_light",
            "max_context": 7800,
            "min_p": 0,
            "math1_temp": 1,
            "math1_quad": 0,
            "math1_quad_entropy_scale": 0
        },
        "Blended-Coffee-Kayra": {
            "order": [
                0,
                1,
                2,
                3
            ],
            "temperature": 1,
            "max_length": 150,
            "min_length": 1,
            "top_k": 25,
            "top_p": 1,
            "tail_free_sampling": 0.925,
            "repetition_penalty": 1.6,
            "repetition_penalty_frequency": 0.001,
            "repetition_penalty_range": 0,
            "repetition_penalty_presence": 0,
            "use_cache": False,
            "return_full_text": False,
            "prefix": "vanilla",
            "phrase_rep_pen": "medium",
            "max_context": 7800,
            "min_p": 0,
            "math1_temp": 1,
            "math1_quad": 0,
            "math1_quad_entropy_scale": 0
        },
        "CosmicCube-Kayra": {
            "order": [
                8,
                5,
                0,
                3
            ],
            "temperature": 0.9,
            "max_length": 150,
            "min_length": 1,
            "typical_p": 0.95,
            "tail_free_sampling": 0.92,
            "mirostat_lr": 0.22,
            "mirostat_tau": 4.95,
            "repetition_penalty": 3,
            "repetition_penalty_range": 4000,
            "repetition_penalty_frequency": 0,
            "repetition_penalty_presence": 0,
            "use_cache": False,
            "return_full_text": False,
            "prefix": "vanilla",
            "phrase_rep_pen": "off",
            "max_context": 7800,
            "min_p": 0,
            "math1_temp": 1,
            "math1_quad": 0,
            "math1_quad_entropy_scale": 0
        },
        "Tea-Time-Kayra": {
            "order": [
                5,
                0,
                4
            ],
            "temperature": 1,
            "max_length": 150,
            "min_length": 1,
            "top_a": 0.017,
            "typical_p": 0.975,
            "repetition_penalty": 3,
            "repetition_penalty_slope": 0.09,
            "repetition_penalty_frequency": 0,
            "repetition_penalty_presence": 0,
            "repetition_penalty_range": 7680,
            "use_cache": False,
            "return_full_text": False,
            "prefix": "vanilla",
            "phrase_rep_pen": "aggressive",
            "max_context": 7800,
            "min_p": 0,
            "math1_temp": 1,
            "math1_quad": 0,
            "math1_quad_entropy_scale": 0
        },
        "Pilotfish-Kayra": {
            "order": [
                0,
                4,
                1,
                2,
                5,
                3
            ],
            "temperature": 1.31,
            "max_length": 150,
            "min_length": 1,
            "top_k": 25,
            "top_p": 0.97,
            "top_a": 0.18,
            "typical_p": 0.98,
            "tail_free_sampling": 1,
            "repetition_penalty": 1.55,
            "repetition_penalty_frequency": 0.00075,
            "repetition_penalty_presence": 0.00085,
            "repetition_penalty_range": 8192,
            "repetition_penalty_slope": 1.8,
            "use_cache": False,
            "return_full_text": False,
            "prefix": "vanilla",
            "phrase_rep_pen": "medium",
            "max_context": 7800,
            "min_p": 0,
            "math1_temp": 1,
            "math1_quad": 0,
            "math1_quad_entropy_scale": 0
        },
        "Green-Active-Writer-Kayra": {
            "order": [
                0,
                8,
                5,
                3
            ],
            "temperature": 1.5,
            "max_length": 150,
            "min_length": 1,
            "typical_p": 0.95,
            "tail_free_sampling": 0.95,
            "mirostat_lr": 0.2,
            "mirostat_tau": 5.5,
            "repetition_penalty": 1,
            "repetition_penalty_range": 1632,
            "repetition_penalty_frequency": 0,
            "repetition_penalty_presence": 0,
            "use_cache": False,
            "return_full_text": False,
            "prefix": "vanilla",
            "phrase_rep_pen": "very_aggressive",
            "max_context": 7800,
            "min_p": 0,
            "math1_temp": 1,
            "math1_quad": 0,
            "math1_quad_entropy_scale": 0
        },
        "Blook-Kayra": {
            "order": [
                2,
                3,
                1,
                0
            ],
            "temperature": 1,
            "max_length": 150,
            "min_length": 1,
            "top_k": 0,
            "top_p": 0.96,
            "tail_free_sampling": 0.96,
            "repetition_penalty": 2,
            "repetition_penalty_slope": 1,
            "repetition_penalty_frequency": 0.02,
            "repetition_penalty_range": 0,
            "repetition_penalty_presence": 0.3,
            "use_cache": False,
            "return_full_text": False,
            "prefix": "vanilla",
            "phrase_rep_pen": "very_aggressive",
            "max_context": 7800,
            "min_p": 0,
            "math1_temp": 1,
            "math1_quad": 0,
            "math1_quad_entropy_scale": 0
        },
        "Pro-Writer-Kayra": {
            "order": [
                3,
                4,
                5,
                0
            ],
            "temperature": 1.06,
            "max_length": 150,
            "min_length": 1,
            "top_a": 0.146,
            "typical_p": 0.976,
            "tail_free_sampling": 0.969,
            "repetition_penalty": 1.86,
            "repetition_penalty_slope": 2.33,
            "repetition_penalty_frequency": 0,
            "repetition_penalty_presence": 0,
            "repetition_penalty_range": 2048,
            "use_cache": False,
            "return_full_text": False,
            "prefix": "vanilla",
            "phrase_rep_pen": "medium",
            "max_context": 7800,
            "min_p": 0,
            "math1_temp": 1,
            "math1_quad": 0,
            "math1_quad_entropy_scale": 0
        },
        "Erato-Shosetsu": {
            "max_context": 8000,
            "temperature": 1,
            "max_length": 150,
            "min_length": 1,
            "top_k": 50,
            "top_p": 0.85,
            "top_a": 1,
            "typical_p": 1,
            "tail_free_sampling": 0.895,
            "repetition_penalty": 1.63,
            "repetition_penalty_range": 1024,
            "repetition_penalty_slope": 3.33,
            "repetition_penalty_frequency": 0.0035,
            "repetition_penalty_presence": 0,
            "phrase_rep_pen": "medium",
            "mirostat_lr": 1,
            "mirostat_tau": 0,
            "math1_temp": 0.3,
            "math1_quad": 0.0645,
            "math1_quad_entropy_scale": 0.05,
            "min_p": 0.05,
            "order": [
                9,
                10
            ]
        },
        "Carefree-Kayra": {
            "order": [
                2,
                3,
                0,
                4,
                1
            ],
            "temperature": 1.35,
            "max_length": 150,
            "min_length": 1,
            "top_k": 15,
            "top_p": 0.85,
            "top_a": 0.1,
            "tail_free_sampling": 0.915,
            "repetition_penalty": 2.8,
            "repetition_penalty_range": 2048,
            "repetition_penalty_slope": 0.02,
            "repetition_penalty_frequency": 0.02,
            "repetition_penalty_presence": 0,
            "use_cache": False,
            "return_full_text": False,
            "prefix": "vanilla",
            "phrase_rep_pen": "aggressive",
            "max_context": 7800,
            "min_p": 0,
            "math1_temp": 1,
            "math1_quad": 0,
            "math1_quad_entropy_scale": 0
        },
        "Erato-Zany-Scribe": {
            "max_context": 8000,
            "temperature": 1,
            "max_length": 150,
            "min_length": 1,
            "top_k": 0,
            "top_p": 0.99,
            "top_a": 1,
            "typical_p": 1,
            "tail_free_sampling": 0.99,
            "repetition_penalty": 1,
            "repetition_penalty_range": 64,
            "repetition_penalty_slope": 1,
            "repetition_penalty_frequency": 0.75,
            "repetition_penalty_presence": 1.5,
            "phrase_rep_pen": "medium",
            "mirostat_lr": 1,
            "mirostat_tau": 1,
            "math1_temp": -0.4,
            "math1_quad": 0.6,
            "math1_quad_entropy_scale": -0.1,
            "min_p": 0.08,
            "order": [
                9,
                2
            ]
        },
        "Stelenes-Kayra": {
            "order": [
                3,
                0,
                5
            ],
            "temperature": 2.5,
            "max_length": 150,
            "min_length": 1,
            "typical_p": 0.969,
            "tail_free_sampling": 0.941,
            "repetition_penalty": 1,
            "repetition_penalty_range": 1024,
            "repetition_penalty_frequency": 0,
            "repetition_penalty_presence": 0,
            "use_cache": False,
            "return_full_text": False,
            "prefix": "vanilla",
            "phrase_rep_pen": "medium",
            "max_context": 7800,
            "min_p": 0,
            "math1_temp": 1,
            "math1_quad": 0,
            "math1_quad_entropy_scale": 0
        },
        "Erato-Golden-Arrow": {
            "max_context": 8000,
            "temperature": 1,
            "max_length": 150,
            "min_length": 1,
            "top_k": 0,
            "top_p": 0.995,
            "top_a": 1,
            "typical_p": 1,
            "tail_free_sampling": 0.87,
            "repetition_penalty": 1.5,
            "repetition_penalty_range": 2240,
            "repetition_penalty_slope": 1,
            "repetition_penalty_frequency": 0,
            "repetition_penalty_presence": 0,
            "phrase_rep_pen": "light",
            "mirostat_lr": 1,
            "mirostat_tau": 0,
            "math1_temp": 0.3,
            "math1_quad": 0.19,
            "math1_quad_entropy_scale": 0,
            "min_p": 0,
            "order": [
                9,
                2
            ]
        }
    }