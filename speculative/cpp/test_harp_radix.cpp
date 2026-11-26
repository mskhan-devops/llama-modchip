#include "rc_harp_radix.h"
#include <cassert>
#include <cstdio>
#include <vector>

int main() {
    auto fp1 = rc_harp_fingerprint::build("modelA.gguf", 32768, 8, 2);
    auto fp2 = rc_harp_fingerprint::build("modelB.gguf", 32768, 8, 2);
    auto fp3 = rc_harp_fingerprint::build("modelA.gguf", 32768, 1, 1);
    assert(fp1.hash != fp2.hash);
    assert(fp1.hash != fp3.hash);

    std::vector<llama_token> blkA(RC_HARP_BLOCK_TOKS), blkB(RC_HARP_BLOCK_TOKS), blkC(RC_HARP_BLOCK_TOKS), blkD(RC_HARP_BLOCK_TOKS);
    for (size_t i = 0; i < RC_HARP_BLOCK_TOKS; ++i) {
        blkA[i] = (llama_token)(1000 + i);
        blkB[i] = (llama_token)(2000 + i);
        blkC[i] = (llama_token)(3000 + i);
        blkD[i] = (llama_token)(4000 + i);
    }

    auto concat = [](const std::vector<llama_token> & x, const std::vector<llama_token> & y) {
        std::vector<llama_token> out = x;
        out.insert(out.end(), y.begin(), y.end());
        return out;
    };

    auto AC = concat(blkA, blkC);
    auto BC = concat(blkB, blkC);
    auto AD = concat(blkA, blkD);

    // 1. Causal divergence: [A, C] and [B, C] MUST have different chained hashes for block 2 (C)
    auto chAC = rc_harp_compute_chain(fp1.hash, AC.size(), [&](size_t i){ return AC[i]; });
    auto chBC = rc_harp_compute_chain(fp1.hash, BC.size(), [&](size_t i){ return BC[i]; });
    assert(chAC.size() == 2 && chBC.size() == 2);
    assert(chAC[0].second != chBC[0].second);
    assert(chAC[1].second != chBC[1].second);

    // 2. Radix index deepest match + branching + clean eviction
    rc_harp_radix_index idx;
    idx.set_seed(fp1.hash);
    idx.insert(10, AC.size(), [&](size_t i){ return AC[i]; });
    idx.insert(20, AD.size(), [&](size_t i){ return AD[i]; });

    // Query [A, C] -> must hit exact deepest match entry 10 ONLY
    auto hitAC = idx.lookup_deepest(AC.size(), [&](size_t i){ return AC[i]; });
    assert(hitAC.size() == 1 && hitAC[0] == 10);

    // Query [A, B] -> must fall back to shared ancestor block A (both 10 and 20)
    auto AB = concat(blkA, blkB);
    auto hitAB = idx.lookup_deepest(AB.size(), [&](size_t i){ return AB[i]; });
    assert(hitAB.size() == 2);

    // Query [B, C] -> MUST return EMPTY (0 false matches on block C!)
    auto hitBC = idx.lookup_deepest(BC.size(), [&](size_t i){ return BC[i]; });
    assert(hitBC.empty());

    // Evict entry 10 -> [A, C] now matches ancestor A via entry 20; evict 20 -> completely empty
    idx.erase(10);
    auto hitAC2 = idx.lookup_deepest(AC.size(), [&](size_t i){ return AC[i]; });
    assert(hitAC2.size() == 1 && hitAC2[0] == 20);
    idx.erase(20);
    assert(idx.hash_to_ids.empty() && idx.id_to_hashes.empty());

    printf("test_harp_radix: ALL ASSERTIONS PASSED (ASan+UBSan clean)!\n");
    return 0;
}
