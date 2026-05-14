#pragma once
#include <memory>
#include <string>
#include <vector>

#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>
#include <GraphMol/Fingerprints/MorganFingerprints.h>
#include <DataStructs/ExplicitBitVect.h>

#include "tsv_utils.hpp"

namespace bscs {

struct FpConfig {
    int radius = 3;
    int nbits  = 4096;
};

// Parse SMILES (handles CXSMILES). Returns nullptr on failure.
inline std::unique_ptr<RDKit::ROMol> smiles_to_mol(const std::string& smiles) {
    std::string clean = strip_cxsmiles(smiles);
    if (clean.empty()) return nullptr;
    try {
        RDKit::ROMol* m = RDKit::SmilesToMol(clean);
        return std::unique_ptr<RDKit::ROMol>(m);
    } catch (...) {
        return nullptr;
    }
}

// Canonical SMILES from mol.
inline std::string mol_to_canonical_smiles(const RDKit::ROMol& mol) {
    try { return RDKit::MolToSmiles(mol); }
    catch (...) { return {}; }
}

// Compute Morgan fingerprint as float vector (bit = 0.0 or 1.0).
inline bool mol_to_fp_float(const RDKit::ROMol& mol, const FpConfig& cfg,
                             float* out) {
    try {
        std::unique_ptr<ExplicitBitVect> fp(
            RDKit::MorganFingerprints::getFingerprintAsBitVect(
                mol, cfg.radius, cfg.nbits));
        if (!fp) return false;
        for (int i = 0; i < cfg.nbits; i++)
            out[i] = fp->getBit(i) ? 1.0f : 0.0f;
        return true;
    } catch (...) {
        return false;
    }
}

// Compute Morgan fingerprint as ExplicitBitVect (for similarity reranking).
inline std::unique_ptr<ExplicitBitVect> mol_to_fp_bv(
        const RDKit::ROMol& mol, const FpConfig& cfg) {
    try {
        return std::unique_ptr<ExplicitBitVect>(
            RDKit::MorganFingerprints::getFingerprintAsBitVect(
                mol, cfg.radius, cfg.nbits));
    } catch (...) {
        return nullptr;
    }
}

// Convenience: SMILES → float vector. Returns false on failure.
inline bool smiles_to_fp_float(const std::string& smiles, const FpConfig& cfg,
                                float* out) {
    auto mol = smiles_to_mol(smiles);
    if (!mol) return false;
    return mol_to_fp_float(*mol, cfg, out);
}

} // namespace bscs
