#ifndef __CUDACRYPTONIGHTR_GEN_H__
#define __CUDACRYPTONIGHTR_GEN_H__

#include <stdint.h>
#include <vector>
#include <string>
#include "common/xmrig.h"

void CryptonightR_get_program(std::vector<char>& ptx, std::string& lowered_name, xmrig::Variant variant, uint64_t height, int arch_major, int arch_minor, bool background = false);

#endif
