#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <iomanip>
#include <cstdint>
#include <cmath>

#include <faiss/IndexIVF.h>
#include <faiss/index_io.h>

#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/Fingerprints/MorganFingerprints.h>
#include <DataStructs/ExplicitBitVect.h>
#include <DataStructs/BitOps.h>

using namespace std;
using namespace RDKit;

enum SimilarityType {
    TANIMOTO,
    DICE,
    TVERSKY,
    COSINE,
    KULCZYNSKI
};

SimilarityType parse_metric(const string& m) {
    if (m == "dice") return DICE;
    if (m == "tversky") return TVERSKY;
    if (m == "cosine") return COSINE;
    if (m == "kulczynski") return KULCZYNSKI;
    return TANIMOTO; // default
}

double compute_cosine(const ExplicitBitVect& a, const ExplicitBitVect& b, int dim) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < dim; i++) {
        int va = a.getBit(i);
        int vb = b.getBit(i);
        dot += va * vb;
        na += va;
        nb += vb;
    }
    return dot / (sqrt(na) * sqrt(nb) + 1e-8);
}

double compute_similarity(SimilarityType type,
                          const ExplicitBitVect& qfp,
                          const ExplicitBitVect& fp,
                          int dim) {

    switch (type) {
        case DICE:
            return DiceSimilarity(qfp, fp);
        case TVERSKY:
            return TverskySimilarity(qfp, fp, 0.7, 0.3);
        case COSINE:
            return compute_cosine(qfp, fp, dim);
        case KULCZYNSKI:
            return KulczynskiSimilarity(qfp, fp);
        case TANIMOTO:
        default:
            return TanimotoSimilarity(qfp, fp);
    }
}

static inline void fp_to_float_inplace(const ExplicitBitVect* fp, float* out, int dim) {
    std::fill(out, out + dim, 0.0f);
    for (int i = 0; i < dim; i++) {
        if (fp->getBit(i)) out[i] = 1.0f;
    }
}

string read_smiles_by_id(ifstream& smiles_file, ifstream& offset_file, int id) {
    uint64_t offset = 0;

    offset_file.clear();
    offset_file.seekg((uint64_t)id * sizeof(uint64_t), ios::beg);
    offset_file.read(reinterpret_cast<char*>(&offset), sizeof(uint64_t));

    smiles_file.clear();
    smiles_file.seekg(offset, ios::beg);

    string line;
    getline(smiles_file, line);
    return line;
}

struct ResultItem {
    int id;
    float faiss_distance;
    double score;
    string smiles;
};

int main(int argc, char* argv[]) {
    cout << " SEARCH + RERANK + OFFSET MODE\n";

    // DEFAULT VALUES 
    string metric_name = "tanimoto";
    int candidate_k = 5000;
    int nprobe = 32;
    size_t max_codes = 0;

    // 🔥 NEW PARAMS
    int radius = 2;
    int nbits_fp = 2048;

    // CLI parsing
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];

        if (arg == "--metric" && i + 1 < argc) {
            metric_name = argv[++i];
        }
        else if (arg == "--k" && i + 1 < argc) {
            candidate_k = stoi(argv[++i]);
        }
        else if (arg == "--nprobe" && i + 1 < argc) {
            nprobe = stoi(argv[++i]);
        }
        else if (arg == "--max_codes" && i + 1 < argc) {
            max_codes = stoull(argv[++i]);
        }
        else if (arg == "--radius" && i + 1 < argc) {
            radius = stoi(argv[++i]);
        }
        else if (arg == "--nbits" && i + 1 < argc) {
            nbits_fp = stoi(argv[++i]);
        }
    }

    const int DIM = nbits_fp;

    SimilarityType sim_type = parse_metric(metric_name);

    cout << " Metric: " << metric_name << "\n";
    cout << " Candidate K: " << candidate_k << "\n";
    cout << " nprobe: " << nprobe << "\n";
    cout << " max_codes: " << max_codes << "\n";
    cout << " FP radius: " << radius << "\n";
    cout << " FP bits: " << nbits_fp << "\n";

    const string smiles_path = "../data/tam.smi";
    const string offset_path = "../index/tam.offsets";
    const string index_path  = "../index/tam_ivfpq_stream.index";

    ifstream smiles_file(smiles_path);
    ifstream offset_file(offset_path, ios::binary);

    if (!smiles_file.is_open() || !offset_file.is_open()) {
        cerr << "Errro File could not be opened.\n";
        return 1;
    }

    faiss::Index* base_index = faiss::read_index(index_path.c_str());
    if (!base_index) {
        cerr << "Error Failed to read the index.\n";
        return 1;
    }

    auto* index = dynamic_cast<faiss::IndexIVF*>(base_index);
    if (index) {
        index->nprobe = nprobe;
        if (max_codes > 0) index->max_codes = max_codes;

        cout << " nprobe = " << index->nprobe << "\n";
        cout << " max_codes = " << index->max_codes << "\n";
    }

    ofstream result_file("result.txt");

    while (true) {
        string query_smiles;
        cout << "\nSMILES (for exit write exit :) ): ";
        cin >> query_smiles;

        if (query_smiles == "exit") break;

        unique_ptr<ROMol> qmol(SmilesToMol(query_smiles));
        if (!qmol) {
            cout << " invalid SMILES\n";
            continue;
        }

        unique_ptr<ExplicitBitVect> qfp(
            MorganFingerprints::getFingerprintAsBitVect(*qmol, radius, nbits_fp)
        );

        vector<float> qvec(DIM, 0.0f);
        fp_to_float_inplace(qfp.get(), qvec.data(), DIM);

        vector<float> D(candidate_k);
        vector<faiss::idx_t> I(candidate_k);

        base_index->search(1, qvec.data(), candidate_k, D.data(), I.data());

        vector<ResultItem> reranked;
        reranked.reserve(candidate_k);

        for (int i = 0; i < candidate_k; i++) {
            int id = static_cast<int>(I[i]);
            if (id < 0) continue;

            string hit_smiles = read_smiles_by_id(smiles_file, offset_file, id);
            if (hit_smiles.empty()) continue;

            unique_ptr<ROMol> mol(SmilesToMol(hit_smiles));
            if (!mol) continue;

            unique_ptr<ExplicitBitVect> fp(
                MorganFingerprints::getFingerprintAsBitVect(*mol, radius, nbits_fp)
            );

            double score = compute_similarity(sim_type, *qfp, *fp, DIM);

            reranked.push_back({
                id,
                D[i],
                score,
                hit_smiles
            });
        }

        sort(reranked.begin(), reranked.end(),
             [](const ResultItem& a, const ResultItem& b) {
                 if (a.score != b.score) return a.score > b.score;
                 return a.faiss_distance < b.faiss_distance;
             });

        cout << "\nTOP RESULTS:\n";
        cout << fixed << setprecision(4);

        result_file << "QUERY: " << query_smiles << "\n";

        for (int i = 0; i < reranked.size(); i++) {
            result_file << i + 1
                        << ". ID=" << reranked[i].id
                        << " | score=" << reranked[i].score
                        << " | faiss_dist=" << reranked[i].faiss_distance
                        << " | SMILES=" << reranked[i].smiles
                        << " | radius=" << radius
                        << " | nbits=" << nbits_fp
                        << "\n";
        }

        int top_n = min(10, (int)reranked.size());
        for (int i = 0; i < top_n; i++) {
            cout << i + 1
                 << ". ID=" << reranked[i].id
                 << " | score=" << reranked[i].score
                 << " | faiss_dist=" << reranked[i].faiss_distance
                 << " | SMILES=" << reranked[i].smiles
                 << " | radius=" << radius
                 << " | nbits=" << nbits_fp
                 << "\n";
        }

        result_file << "----------------------\n";
    }

    delete base_index;
    return 0;
}