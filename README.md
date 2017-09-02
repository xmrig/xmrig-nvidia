# XMRig
XMRig is high performance Monero (XMR) NVIDIA miner, with the official full Windows support.

GPU mining part based on [psychocrypt](https://github.com/psychocrypt) code used in xmr-stak-nvidia.

* This is the NVIDIA GPU mining version, there is also a [CPU version](https://github.com/xmrig/xmrig).

<img src="https://i.imgur.com/wRCZ3IJ.png" width="620" >

#### Table of contents
* [Features](#features)
* [Download](#download)
* [Usage](#usage)
* [Build](https://github.com/xmrig/xmrig-nvidia/wiki/Build)
* [Donations](#donations)
* [Contacts](#contacts)

## Features
* High performance.
* Official Windows support.
* Support for backup (failover) mining server.
* CryptoNight-Lite support for AEON.
* Automatic GPU configuration.
* GPU health monitoring (clocks, power, temperature, fan speed) 
* Nicehash support.
* It's open source software.

## Download
* Binary releases: https://github.com/xmrig/xmrig-nvidia/releases
* Git tree: https://github.com/xmrig/xmrig-nvidia.git
  * Clone with `git clone https://github.com/xmrig/xmrig-nvidia.git`  :hammer: [Build instructions](https://github.com/xmrig/xmrig-nvidia/wiki/Build).

## Usage

### Command line options
```
  -a, --algo=ALGO         cryptonight (default) or cryptonight-lite
  -o, --url=URL           URL of mining server
  -O, --userpass=U:P      username:password pair for mining server
  -u, --user=USERNAME     username for mining server
  -p, --pass=PASSWORD     password for mining server
  -k, --keepalive         send keepalived for prevent timeout (need pool support)
  -r, --retries=N         number of times to retry before switch to backup server (default: 5)
  -R, --retry-pause=N     time to pause between retries (default: 5)
      --no-color          disable colored output
      --donate-level=N    donate level, default 5% (5 minutes in 100 minutes)
      --user-agent        set custom user-agent string for pool
  -B, --background        run the miner in the background
  -c, --config=FILE       load a JSON-format configuration file
  -l, --log-file=FILE     log all output to a file
      --nicehash          enable nicehash support
      --print-time=N      print hashrate report every N seconds
  -h, --help              display this help and exit
  -V, --version           output version information and exit

Auto-configuration specific options:
      --bfactor=[0-12]    run CryptoNight core kernel in smaller pieces
                          from 0 (ui freeze) to 12 (smooth), Windows default is 6
      --bsleep=N          insert a delay of N microseconds between kernel launches
      --max-gpu-threads=N limit maximum count of GPU threads
```

### Config file.
GPU configuration now possible only via config file. Sample config:
```json
{
    "background": false,
    "colors": true,
    "donate-level": 5,
    "log-file": null,
    "print-time": 60,
    "retries": 5,
    "retry-pause": 5,
    "syslog": false,
    "threads": [
        {
            "index": 0,
            "threads": 42,
            "blocks": 18,
            "bfactor": 6,
            "bsleep": 25
        }
    ],
    "pools": [
        {
            "url": "pool.minemonero.pro:5555",
            "user": "",
            "pass": "x",
            "keepalive": true,
            "nicehash": false
        }
    ]
}
```
If `threads` option not specified the miner will try automatically create optimal configuration for your GPUs.

## Donations
Default donation 5% (5 minutes in 100 minutes) can be reduced to 1% via command line option `--donate-level`.

* XMR: `48edfHu7V9Z84YzzMa6fUueoELZ9ZRXq9VetWzYGzKt52XU5xvqgzYnDK9URnRoJMk1j8nLwEVsaSWJ4fhdUyZijBGUicoD`
* BTC: `1P7ujsXeX7GxQwHNnJsRMgAdNkFZmNVqJT`

## Contacts
* support@xmrig.com
* [reddit](https://www.reddit.com/user/XMRig/)
