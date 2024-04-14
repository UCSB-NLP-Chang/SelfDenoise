# This is a very simple cache that stores the openAI API responses in jsonl files.

import os
from os import path
from pathlib import Path
import portalocker

import hashlib
import json
import linecache
from typing import Any, Dict, List

KEY_NAMES = [
    "model",
    "seed",
    "top_p",
    "temperature",
    "messages",
]

def gen_md5(key_str):
    m = hashlib.md5()
    m.update(key_str.encode('utf-8'))
    return str(m.hexdigest()[0:16])

class SimpleCache:
    def __init__(self, cache_dir, max_in_mem_cache=10) -> None:
        self.cache_dir = Path(cache_dir)
        self.max_in_mem_cache = max_in_mem_cache
        self.init()
    
    @property
    def key_path(self):
        return Path(self.cache_dir) / "keys.json"
    
    @property
    def cache_path(self):
        return Path(self.cache_dir) / "cache.jsonl"
    
    def key_hanlder(self, mode='r'):
        f = open(self.key_path, mode=mode)
        return f
    
    def cache_handler(self, mode='r'):
        f = open(self.cache_path, mode=mode)
        return f
    
    @property
    def max_line_no(self):
        full_list = list(self.in_mem_keys.values()) + list(self.full_keys.values())
        if len(full_list) == 0:
            return 1
        else:
            return max(full_list) + 1
    
    def reload_full_keys(self):
        with portalocker.Lock(self.key_path, mode='r+', timeout=60) as fh:
            self.full_keys = json.load(fh)
   
    def init(self):
        if not path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        if not path.exists(self.key_path):
            with self.key_hanlder(mode="w") as f:
                json.dump({}, f, indent=2) # create empty cache

        if not path.exists(self.cache_path):
            with self.cache_handler(mode="w") as _:
                pass

        self.reload_full_keys()
        self.in_mem_keys = {}
        self.in_mem_cache = {}
        linecache.clearcache()

    def gen_key_str(self, query):
        key_str = []
        for key, value in sorted(query.items()):
            if key not in KEY_NAMES:
                continue
            if key == "messages":
                key_str.append("{}={}".format(
                    key, gen_md5(str(value))
                ))
            else:
                key_str.append("{}={}".format(key, value))
        key_str = "-".join(key_str)
        return key_str
    
    def fetch(self, query):
        query_str = self.gen_key_str(query)
        query_key = gen_md5(query_str)
        if query_key in self.in_mem_cache:
            return self.in_mem_cache[query_key]
        elif query_key in self.full_keys:
            line = linecache.getline(self.cache_path.as_posix(), self.full_keys[query_key])
            line = json.loads(line.strip())
            return line
        else:
            return None
        
    def can_fetch(self, query):
        return self.fetch(query) is not None

    #! cache one api json response
    def __call__(self, query, response):
        try_fetch = self.fetch(query)
        if try_fetch is None:
            query_key = gen_md5(self.gen_key_str(query))
            response['unique_key'] = query_key
            self.in_mem_keys[query_key] = self.max_line_no
            self.in_mem_cache[query_key] = response
            if len(self.in_mem_cache) >= self.max_in_mem_cache:
                self.dump()

    def dump(self):
        self.reload_full_keys()
        # with self.cache_handler('a') as f:
        with portalocker.Lock(self.cache_path, mode='a', timeout=60) as f:
            for unique_key, response in self.in_mem_cache.items():
                f.write(json.dumps(response)+"\n")
                self.full_keys[unique_key] = self.in_mem_keys[unique_key]
        
        with portalocker.Lock(self.key_path, mode='w', timeout=60) as f:
            json.dump(self.full_keys, f, indent=2)
        self.init()

def cache_api_call(cacher):
    def decorator(api_call_func):
        def wrapper(*args, **kwargs):
            if cacher.can_fetch(kwargs):
                response = cacher.fetch(kwargs)
            else:
                response = api_call_func(*args, **kwargs)
                cacher(kwargs, response)
            return response 
        return wrapper
    return decorator


APICacher = SimpleCache(
    os.environ.get("API_CACHE_PATH", "./apicache/"),
    10,
)