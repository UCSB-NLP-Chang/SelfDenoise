# This file implments a custom logger that can save logs as jsonl files utilizing structlog library.

import logging
import logging.config
from typing import Any
import structlog
from structlog import DropEvent

class Dropper:
    def __init__(self, dropkey, dropval):
        self._dropkey = dropkey
        self._dropval = dropval

    def __call__(self, logger, method_name, event_dict):
        if event_dict.get(self._dropkey) == self._dropval:
            raise DropEvent

        return event_dict

def drop_httpx(logger, method_name, event_dict):
    print(event_dict)
    if logger == "httpx":
        raise DropEvent

    return event_dict

def configure_structlog(filename):

    # pre_chain is used for calls to the python logging library
    # ex: logging.getLogger().info('some event')
    pre_chain = [
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_logger_name
    ]

    logging_config = {
        'version': 1,
        'handlers': {
            'structured_console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'structured',
            },
            'json_file': {
                'class': 'logging.FileHandler',
                'filename': filename,
                'level': 'INFO',
                'formatter': 'json'
            },
            "debug_console" : {
                'class': 'logging.StreamHandler',
                'level': 'DEBUG',
                'formatter': 'structured',
            }
        },
        'formatters': {
            'json': {
                '()': structlog.stdlib.ProcessorFormatter,
                "processor": structlog.processors.JSONRenderer(),
                'foreign_pre_chain': pre_chain
            },
            'structured': {
                '()': structlog.stdlib.ProcessorFormatter,
                'processor': structlog.dev.ConsoleRenderer(colors=True),
                'foreign_pre_chain': pre_chain
            },
        },
        'loggers': {
            '': {
                'handlers': ['structured_console', 'json_file'],
                'level': 'INFO',
                'propagate': False,
            },
            'debug' : {
                'handlers' : ['debug_console'],
                'level' : 'DEBUG',
                'formatter' : 'structured',
            }
        }
    }

    logging.config.dictConfig(logging_config)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_logger_name,
            drop_httpx,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
