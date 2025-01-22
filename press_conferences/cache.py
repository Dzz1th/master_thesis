import hashlib
import json
import os
import logging
import typing as t


from langchain.globals import set_llm_cache
from langchain.schema.cache import RETURN_VAL_TYPE
from redis import Redis
from langchain_core.caches import BaseCache
from langchain_core.load import dumps, loads
from langchain_core.outputs import Generation

logger = logging.getLogger(__file__)


class LocalRedisCache(BaseCache):
    """Cache that uses Redis as a backend. This module extends LLM Caching
    from langchain library to allow persistent storage.
    """

    def __init__(self, redis_: Redis, persistent_path: os.PathLike = None, ttl: t.Optional[int] = None):
        """Initialize an instance of RedisCache.

        This method initializes an object with Redis caching capabilities.
        It takes a `redis_` parameter, which should be an instance of a Redis
        client class, allowing the object to interact with a Redis
        server for caching purposes.

        Args:
            redis_ (Redis): An instance of a Redis client class
                (e.g., redis.Redis) used for caching.
                This allows the object to communicate with a
                Redis server for caching operations.
            persistent_path (str): Persistent path of cache.
            ttl (t.Optional[int], optional): Time-to-live (TTL) for cached items in seconds.
                If provided, it sets the time duration for how long cached
                items will remain valid. If not provided, cached items will not
                have an automatic expiration.
        """

        if not isinstance(redis_, Redis):
            raise ValueError("Please pass in Redis object.")

        self.persistent_path = persistent_path or get_cache_path('redis_cache')
        self.redis = redis_
        self.ttl = ttl

    def _key(self, prompt: str, llm_string: str) -> str:
        """Compute key from prompt and llm_string"""
        return hashlib.md5(f'{prompt}{llm_string}'.encode()).hexdigest()
    
    def erase_element(self, prompt: str, llm_string: str) -> bool:
        "Erase element based on prompt and llm string"
        key = self._key(prompt, llm_string)
        result = self.redis.delete(key)
        return result > 0

    def lookup(self, prompt: str, llm_string: str) -> t.Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        generations = []
        # Read from a Redis HASH
        results = self.redis.hgetall(self._key(prompt, llm_string))
        if results:
            for _, text in results.items():
                try:
                    generations.append(loads(text))
                except Exception:
                    logger.warning(
                        "Retrieving a cache value that could not be deserialized "
                        "properly. This is likely due to the cache being in an "
                        "older format. Please recreate your cache to avoid this "
                        "error."
                    )
                    # In a previous life we stored the raw text directly
                    # in the table, so assume it's in that format.
                    generations.append(Generation(text=text))
        return generations if generations else None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""
        for gen in return_val:
            if not isinstance(gen, Generation):
                raise ValueError(
                    "RedisCache only supports caching of normal LLM generations, "
                    f"got {type(gen)}"
                )
        # Write to a Redis HASH
        key = self._key(prompt, llm_string)
        mapping = {}
        for idx, generation in enumerate(return_val):
            generation.message.additional_kwargs = {} # We erase additional_kwargs to eliminate not-implemented issue with build-in-cache serialization of schemas.
            dumped = dumps(generation)
            mapping[idx] = dumped

        with self.redis.pipeline() as pipe:
            pipe.hset(
                key,
                mapping=mapping,
            )
            if self.ttl is not None:
                pipe.expire(key, self.ttl)

            pipe.execute()

    def clear(self, **kwargs: t.Any) -> None:
        """Clear cache. If `asynchronous` is True, flush asynchronously."""
        asynchronous = kwargs.get("asynchronous", False)
        self.redis.flushdb(asynchronous=asynchronous, **kwargs)

    def dump_to_disk(self) -> None:
        """Dump the cache contents to a file."""
        cache_contents = {}
        for key in self.redis.scan_iter("*"):
            # Decode the key to a string
            str_key = key.decode("utf-8")
            values = self.redis.hgetall(key)
            decoded_values = {k.decode("utf-8"): v.decode("utf-8") for k, v in values.items()}
            cache_contents[str_key] = decoded_values

        # Serialize and save to a file
        with open(self.persistent_path, "w") as file:
            json.dump(cache_contents, file)

    def load_from_disk(self) -> None:
        """Load cache contents from a file."""
        if os.path.exists(self.persistent_path):
            with open(self.persistent_path, "r") as file:
                cache_contents = json.load(file)

            with self.redis.pipeline() as pipe:
                for key, values in cache_contents.items():
                    pipe.hmset(key, values)
                    if self.ttl is not None:
                        pipe.expire(key, self.ttl)
                pipe.execute()


def set_cache():
    r = LocalRedisCache(Redis.from_url("redis://localhost:6379"),
                                persistent_path='/Users/dzz1th/Job/mgi/Soroka/llm_cache.json')
            
    r.load_from_disk()
    set_llm_cache(r)