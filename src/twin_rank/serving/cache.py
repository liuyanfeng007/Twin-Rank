"""推荐结果缓存：优先 Redis，不可用时自动退回进程内 TTL 缓存（连接失败不会拖垮服务）。
"""
import json
import logging
import time

logger = logging.getLogger(__name__)


class MemoryCache:
    def __init__(self, ttl: int):
        self.ttl, self.data = ttl, {}

    def get(self, key):
        hit = self.data.get(key)
        if hit and hit[0] > time.time():
            return hit[1]
        self.data.pop(key, None)
        return None

    def set(self, key, value) -> None:
        self.data[key] = (time.time() + self.ttl, value)

    def delete_prefix(self, prefix: str) -> None:
        for key in [k for k in self.data if k.startswith(prefix)]:
            del self.data[key]


class RedisCache:
    def __init__(self, url: str, ttl: int):
        import redis
        self.client, self.ttl = redis.Redis.from_url(url), ttl
        self.client.ping()

    def get(self, key):
        raw = self.client.get(key)
        return json.loads(raw) if raw else None

    def set(self, key, value) -> None:
        self.client.setex(key, self.ttl, json.dumps(value, ensure_ascii=False))

    def delete_prefix(self, prefix: str) -> None:
        for key in self.client.scan_iter(match=f"{prefix}*"):
            self.client.delete(key)


class NoCache:
    def get(self, key): return None
    def set(self, key, value): pass
    def delete_prefix(self, prefix): pass


def make_cache(cfg):
    if not cfg.enabled:
        return NoCache()
    if cfg.redis_url:
        try:
            cache = RedisCache(cfg.redis_url, cfg.ttl_seconds)
            logger.info("使用 Redis 缓存: %s", cfg.redis_url)
            return cache
        except Exception as exc:                       # 连接失败不应拖垮服务
            logger.warning("Redis 不可用(%s)，退回进程内缓存", exc)
    return MemoryCache(cfg.ttl_seconds)
