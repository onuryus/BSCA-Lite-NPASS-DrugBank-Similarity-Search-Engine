#pragma once
#include <string>
#include <unordered_map>
#include <vector>

namespace bscs {

// A single row from any NPASS table: named fields.
using NpassRow = std::unordered_map<std::string, std::string>;

// Full annotation for one NPASS compound.
struct NpassCompound {
    std::string np_id;

    // From NPASS3.0_naturalproducts_structure.txt
    std::string inchi, inchikey, smiles_npass;

    // From NPASS3.0_naturalproducts_generalinfo.txt
    NpassRow generalinfo;

    // From NPASS3.0_activities.txt (one row per activity, joined with target)
    struct ActivityRecord {
        NpassRow activity;  // all columns from activities.txt
        NpassRow target;    // joined target row (from target.txt via target_id)
    };
    std::vector<ActivityRecord> activities;

    // From NPASS3.0_toxicity.txt (same structure)
    struct ToxicityRecord {
        NpassRow toxicity;
        NpassRow target;
    };
    std::vector<ToxicityRecord> toxicities;

    // From NPASS3.0_naturalproducts_species_pair.txt + species_info.txt
    struct SpeciesPairRecord {
        NpassRow species_pair;   // all columns from species_pair.txt
        NpassRow species_info;   // joined species_info row via org_id
    };
    std::vector<SpeciesPairRecord> species_pairs;

    // Contextual biosynthesis records
    std::vector<NpassRow> coculture_records;
    std::vector<NpassRow> elicitation_records;
    std::vector<NpassRow> engineer_records;
    std::vector<NpassRow> symbiont_records;
};

} // namespace bscs
