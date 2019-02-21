import asyncio
import aiohttp
import aiohttp_socks
import aiofiles
import logging
import re
import time
import functools
import random

log = logging.getLogger(__name__)

class CaptchaGrabber:
    def __init__(self, config):
        self._config = config
        self._sess = None

    async def grab(self, count):
        try:
            log.info('targetting url: {}'.format(config['base']))
            params = dict(
                connector=aiohttp_socks.SocksConnector.from_url('socks5://localhost:9150', rdns=True),
                headers={
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:65.0) Gecko/20100101 Firefox/65.0'
                },
            )
            async with aiohttp.ClientSession(**params) as sess:
                async def _retrieved_session_id():
                    log.info('getting session id')
                    began_at = time.time()
                    async with sess.get(self._config['base']) as resp:
                        id = resp.cookies['MARKET_SESSION'].value
                    log.info('got session id: {} ({:.02f} sec.)'.format(id, time.time() - began_at))
                    return id
                session_id = await _retrieved_session_id()

            params = dict(
                connector=aiohttp_socks.SocksConnector.from_url('socks5://localhost:9150', rdns=True, limit=self._config['conns']),
                cookies=dict(MARKET_SESSION='{}'.format(session_id)),
                headers={
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:65.0) Gecko/20100101 Firefox/65.0'
                },
            )
            async with aiohttp.ClientSession(**params) as self._sess:
                return await asyncio.gather(*[self._retrieve(i) for i in range(count)])
        finally:
            self._sess = None

    async def _retrieve(self, nr):
        log.info('{}: fetching'.format(nr))
        began_at = time.time()
        while True:
            try:
                async with self._sess.get('{}/img/captcha3?tcode={}'.format(self._config['base'], random.getrandbits(31))) as resp:
                    async with aiofiles.open('{}-{}.jpg'.format(self._config['prefix'], nr), 'wb') as f:
                        await f.write(await resp.read())
                    log.info('{}: fetched ({:.02f} sec.)'.format(nr, time.time() - began_at))
                    return
            except (aiohttp.ClientResponseError, aiohttp.client_exceptions.ServerDisconnectedError) as e:
                log.warning('{}: connection lost, retrying ({})'.format(nr, e))

if __name__ == '__main__':
    import getopt
    import sys
    import uvloop

    uvloop.install()

    logging.basicConfig(level=logging.INFO, format='%(msg)s')

    config = dict(
        base='http://k3pd243s57fttnpa.onion',
        conns=100, # aiohttp defaults
        prefix='dm',
    )

    opts, base = getopt.gnu_getopt(sys.argv[1:], 'p:t:', ['prefix=', 'threads='])
    for o,a in opts:
        if o in ['-p', '--prefix']:
            config['prefix'] = a
        if o in ['-t', '--threads']:
            config['conns'] = int(a)

    if base:
        config['base'] = base[0]

    began_at = time.time()
    asyncio.run(CaptchaGrabber(config).grab(1000))
    log.info('done ({:.02f} sec.)'.format(time.time() - began_at))
