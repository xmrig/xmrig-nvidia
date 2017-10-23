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
