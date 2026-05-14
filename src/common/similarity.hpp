#pragma once
#include <stdexcept>
#include <string>

#include <DataStructs/ExplicitBitVect.h>
#include <DataStructs/BitOps.h>

namespace bscs {

enum class SimMetric { TANIMOTO, DICE, TVERSKY, COSINE, KULCZYNSKI };

inline SimMetric parse_metric(const std::string& s) {
    if (s == "dice")       return SimMetric::DICE;
    if (s == "tversky")    return SimMetric::TVERSKY;
    if (s == "cosine")     return SimMetric::COSINE;
    if (s == "kulczynski") return SimMetric::KULCZYNSKI;
    if (s == "tanimoto")   return SimMetric::TANIMOTO;
    throw std::invalid_argument("Unknown metric: " + s);
}

inline const char* metric_name(SimMetric m) {
    switch (m) {
        case SimMetric::DICE:       return "dice";
        case SimMetric::TVERSKY:    return "tversky";
        case SimMetric::COSINE:     return "cosine";
        case SimMetric::KULCZYNSKI: return "kulczynski";
        default:                    return "tanimoto";
    }
}

inline double compute_similarity(SimMetric type,
                                  const ExplicitBitVect& q,
                                  const ExplicitBitVect& h) {
    switch (type) {
        case SimMetric::DICE:       return DiceSimilarity(q, h);
        case SimMetric::TVERSKY:    return TverskySimilarity(q, h, 0.7, 0.3);
        case SimMetric::COSINE:     return CosineSimilarity(q, h);
        case SimMetric::KULCZYNSKI: return KulczynskiSimilarity(q, h);
        default:                    return TanimotoSimilarity(q, h);
    }
}

} // namespace bscs
