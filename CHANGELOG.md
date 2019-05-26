# v2.14.4
- [#260](https://github.com/xmrig/xmrig-nvidia/issues/260) Fixed `cn/r` algorithm slowdown with CUDA 10.1+.
- [#268](https://github.com/xmrig/xmrig-nvidia/pull/268) Added support for NVIDIA Jetson.
- In HTTP API for unknown hashrate now used `null` instead of `0.0`.
- Fixed MSVC 2019 version detection.
- Removed obsolete automatic variants.
 
# v2.14.3
- [#260](https://github.com/xmrig/xmrig-nvidia/issues/260) Added workaround for CUDA bug, thanks [@psychocrypt](https://github.com/psychocrypt).

# v2.14.2
- [#253](https://github.com/xmrig/xmrig-nvidia/pull/253) Fixed NVRTC dll copy when build miner.
- [#255](https://github.com/xmrig/xmrig-nvidia/pull/255) Fixed CUDA8 support and added memory size display in summary.

# v2.14.1
- [#246](https://github.com/xmrig/xmrig-nvidia/issues/246) Fixed compatibility with old GPUs (compute capability < 3.5).

# v2.14.0
- **[#242](https://github.com/xmrig/xmrig-nvidia/pull/242) Added new algorithm `cryptonight/rwz`, short alias `cn/rwz` (also known as CryptoNight ReverseWaltz), for upcoming [Graft](https://www.graft.network/) fork.**
- **[#931](https://github.com/xmrig/xmrig/issues/931) Added new algorithm `cryptonight/zls`, short alias `cn/zls` for [Zelerius Network](https://zelerius.org) fork.**
- **[#940](https://github.com/xmrig/xmrig/issues/940) Added new algorithm `cryptonight/double`, short alias `cn/double` (also known as CryptoNight HeavyX), for [X-CASH](https://x-cash.org/).**
- [#241](https://github.com/xmrig/xmrig-nvidia/issues/241) Minimum required CUDA version increased to 8.0 due CUDA 7.5 lacks support for `nvrtcAddNameExpression`.
- [#951](https://github.com/xmrig/xmrig/issues/951#issuecomment-469581529) Fixed crash if AVX was disabled on OS level.
- [#952](https://github.com/xmrig/xmrig/issues/952) Fixed compile error on some Linux.
- [#957](https://github.com/xmrig/xmrig/issues/957#issuecomment-468890667) Added support for embedded config.

# v2.13.0
- **[#938](https://github.com/xmrig/xmrig/issues/938) Added support for new algorithm `cryptonight/r`, short alias `cn/r` (also known as CryptoNightR or CryptoNight variant 4), for upcoming [Monero](https://www.getmonero.org/) fork on March 9, thanks [@SChernykh](https://github.com/SChernykh).**
- [#939](https://github.com/xmrig/xmrig/issues/939) Added support for dynamic (runtime) pools reload.

# v2.12.0-beta
- [#235](https://github.com/xmrig/xmrig-nvidia/pull/235) Added support for new algorithm `cryptonight/wow`, short alias `cn/wow` (also known as CryptonightR), for upcoming [Wownero](http://wownero.org) fork on February 14.

# v2.11.0
- [#928](https://github.com/xmrig/xmrig/issues/928) Added support for new algorithm `cryptonight/gpu`, short alias `cn/gpu` (original name `cryptonight-gpu`), for upcoming [Ryo currency](https://ryo-currency.com) fork on February 14.

# v2.10.0
- [#904](https://github.com/xmrig/xmrig/issues/904) Added new algorithm `cn-pico/trtl` (aliases `cryptonight-turtle`, `cn-trtl`) for upcoming TurtleCoin (TRTL) fork.

# v2.9.4
- [#913](https://github.com/xmrig/xmrig/issues/913) Fixed Masari (MSR) support (this update required for upcoming fork).

# v2.9.3
- [#232](https://github.com/xmrig/xmrig-nvidia/pull/232) Fixed `cn/half` compute errors.

# v2.9.1
- [#899](https://github.com/xmrig/xmrig/issues/899) Added support for new algorithm `cn/half` for Masari and Stellite forks.

# v2.8.4
- Improved `cn/2` performance especially for old GPUs.
- Better `cn/2` autoconfig for old GPUs if variant `-1` or `2` selected.

# v2.8.3
- [#197](https://github.com/xmrig/xmrig/issues/197) Fixed wrong default value for option `sync_mode`.
- [#813](https://github.com/xmrig/xmrig/issues/813) Fixed critical bug with Minergate pool and variant 2.

# v2.8.1
- [#167](https://github.com/xmrig/xmrig-amd/issues/167) Fixed wrong hashrate in `GET /1/threads` endpoint.
- [#204](https://github.com/xmrig/xmrig-nvidia/issues/204) Fixed regression, periodical health reports was not shown since v2.7.0-beta.

# v2.8.0
- **[#753](https://github.com/xmrig/xmrig/issues/753) Added new algorithm [CryptoNight variant 2](https://github.com/xmrig/xmrig/issues/753) for Monero fork, thanks [@SChernykh](https://github.com/SChernykh).**
- **[#758](https://github.com/xmrig/xmrig/issues/758) Added SSL/TLS support for secure connections to pools.**
  - Added per pool options `"tls"` and `"tls-fingerprint"` and command line equivalents.  
- [#245](https://github.com/xmrig/xmrig-proxy/issues/245) Fixed API ID collision when run multiple miners on same machine.
- [#757](https://github.com/xmrig/xmrig/issues/757) Fixed send buffer overflow.
- [#777](https://github.com/xmrig/xmrig/issues/777) Better report about pool connection issues. 

# v2.7.0-beta
- Algorithm variant `cryptonight-lite/ipbc` replaced to `cryptonight-heavy/tube` for **Bittube (TUBE)** coin.
- Added `cryptonight-heavy/xhv` variant for **Haven Protocol (XHV)** coin.
- Added `cryptonight/rto` (cryptonight variant 1 with IPBC/TUBE mod) variant for **Arto (RTO)** coin.
- Added `cryptonight/xao` (original cryptonight with bigger iteration count) variant for **Alloy (XAO)** coin.
- Added `cryptonight/xtl` variant for **Stellite (XTL)** coin.
- Added `cryptonight/msr` also known as `cryptonight-fast` for **Masari (MSR)** coin.
- Added new detailed hashrate report.
- Added command line option `--dry-run`.

# v2.6.1
- Fixed critical bug, in some cases miner was can't recovery connection and switch to failover pool, version 2.5.2 and v2.6.0-beta1 affected.
- [#499](https://github.com/xmrig/xmrig/issues/499) IPv6 support disabled for internal HTTP API.
- Added workaround for nicehash.com if you use `cryptonightv7.<region>.nicehash.com` option `variant=1` will be set automatically.

# v2.6.0-beta1
 - [#476](https://github.com/xmrig/xmrig/issues/476) **Added Cryptonight-Heavy support for Sumokoin and Haven Protocol ASIC resistance fork.**
 - Added short aliases for algorithm names: `cn`, `cn-lite` and `cn-heavy`.
 
# v2.5.2
- [#448](https://github.com/xmrig/xmrig/issues/478) Fixed broken reconnect.

# v2.5.1
- [#454](https://github.com/xmrig/xmrig/issues/454) Fixed build with libmicrohttpd version below v0.9.35.
- [#456](https://github.com/xmrig/xmrig/issues/459) Verbose errors related to donation pool was not fully silenced.
- [#459](https://github.com/xmrig/xmrig/issues/459) Fixed regression (version 2.5.0 affected) with connection to **xmr.f2pool.com**.

# v2.5.0
- [#434](https://github.com/xmrig/xmrig/issues/434) **Added support for Monero v7 PoW, scheduled on April 6.**
- Improved automatic configuration, previous may give not usable suggested config.
- Added full IPv6 support.
- Added protocol extension, when use the miner with xmrig-proxy 2.5+ no more need manually specify `nicehash` option.
- [#51](https://github.com/xmrig/xmrig-amd/issues/51) Fixed multiple pools in initial config was saved incorrectly.
- [#123](https://github.com/xmrig/xmrig-proxy/issues/123) Fixed regression (all versions since 2.4 affected) fragmented responses from pool/proxy was parsed incorrectly.

# v2.4.5
 - [#49](https://github.com/xmrig/xmrig-amd/issues/49) Fixed, in some cases, pause was cause an infinite loop.
 - [#64](https://github.com/xmrig/xmrig-nvidia/issues/64) Fixed compatibility with CUDA 9.1.
 - [#84](https://github.com/xmrig/xmrig-nvidia/issues/84) In some cases miner was doesn't write log to stdout.
 - Added libmicrohttpd version to --version output.
 - Fixed bug in singal handler, in some cases miner wasn't shutdown properly.
 - Fixed recent MSVC 2017 version detection.
 - Fixed, config file options `cuda-bfactor` and `cuda-bsleep` was ignored.

# v2.4.2
 - Added [HTTP API](https://github.com/xmrig/xmrig/wiki/API).
 - Added CUDA 9 and Volta GPUs support.
 - Added support for set CPU affinity for GPU threads via command line option `--cuda-affinity` and config option `affine_to_cpu`.
 - Improved automatic configuration.
 - Added comments support in config file.
 - libjansson replaced to rapidjson.
 - [#11](https://github.com/xmrig/xmrig-nvidia/issues/11#issuecomment-336796627) GPU configuration via command line options now fully supported. Added options `--cuda-devices`, `--cuda-launch`, `--cuda-bfactor` and `--cuda-bsleep`. Options `--bfactor`, `--bsleep`, `--max-gpu-threads` now deprecated.
 - [#17](https://github.com/xmrig/xmrig-nvidia/issues/17) Fixed issues with NVML monitoring API.
 - [#98](https://github.com/xmrig/xmrig/issues/98) Ignore `keepalive` option with minergate.com and nicehash.com.
 - [#101](https://github.com/xmrig/xmrig/issues/101) Fixed MSVC 2017 (15.3) compile time version detection.
 - [#108](https://github.com/xmrig/xmrig/issues/108) Silently ignore invalid values for `donate-level` option.
 - [#147](https://github.com/xmrig/xmrig/issues/147) Fixed comparability with monero-stratum.
 - [#153](https://github.com/xmrig/xmrig/issues/153) Fixed issues with dwarfpool.com.
 
# v2.3.1
- [#7](https://github.com/xmrig/xmrig-nvidia/issues/7) Fixed crash when try see hashrate reports in some configurations.
- [#8](https://github.com/xmrig/xmrig-nvidia/issues/8) Fixed build without AEON support.

# v2.3.0
- [#5](https://github.com/xmrig/xmrig-nvidia/issues/5) Added AEON (cryptonight-lite) support.
- Added GPU health monitoring: clocks, power, temperature and fan speed via NVML API.
- Added options `bfactor`, `bsleep` and `max-gpu-threads` for change auto configuration result.

# v2.3.0-beta2
- Less aggressive auto-configuration. Solved wrong configuration with some GPUs.

# v2.3.0-beta1
- First public release.
