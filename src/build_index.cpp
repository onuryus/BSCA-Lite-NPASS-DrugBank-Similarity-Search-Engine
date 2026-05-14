#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <memory>
#include <algorithm>
#include <cstdint>
#include <iomanip>

#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>

#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/Fingerprints/MorganFingerprints.h>
#include <DataStructs/ExplicitBitVect.h>

#include <omp.h>

using namespace std;
using namespace RDKit;

// fingerprint -> float
static inline void fp_to_float_inplace(const ExplicitBitVect* fp, float* out, int dim) {
    std::fill(out, out + dim, 0.0f);
    for (int i = 0; i < dim; i++) {
        if (fp->getBit(i)) out[i] = 1.0f;
    }
}

int main(int argc, char* argv[]) {

    cout << " STARTED (ULTRA OPT STREAMING + ETA)\n";

    // ===== CLI PARAMS =====
    int radius = 2;
    int nbits_fp = 2048;
    size_t batch_size = 20000;  

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];

        if (arg == "--radius" && i + 1 < argc)
            radius = stoi(argv[++i]);

        else if (arg == "--nbits" && i + 1 < argc)
            nbits_fp = stoi(argv[++i]);

        else if (arg == "--batch" && i + 1 < argc)
            batch_size = stoull(argv[++i]);
    }

    cout << " FP radius: " << radius << "\n";
    cout << " FP bits: " << nbits_fp << "\n";
    cout << " Batch size: " << batch_size << "\n";

    const int DIM = nbits_fp;

    const string input_file = "../data/tam.smi";
    const string out_index = "../index/tam_ivfpq_stream.index";
    const string out_offsets = "../index/tam.offsets";

    const int nlist = 1024;
    const int m = 16;
    const int nbits = 8;

    const size_t train_target = 50000;

    system("mkdir -p ../index");

    // ERROR FILE
    ofstream error_out("../index/error.smi");

    auto global_start = chrono::high_resolution_clock::now();

    // ---------- TOTAL COUNT ----------
    ifstream fin_count(input_file);
    size_t total_lines = 0;
    string line;

    while (getline(fin_count, line)) {
        if (!line.empty()) total_lines++;
    }

    cout << " Total molecules: " << total_lines << "\n";

    // ---------- TRAIN ----------
    ifstream fin_train(input_file);
    vector<string> train_smiles;

    while (getline(fin_train, line) && train_smiles.size() < train_target) {
        if (!line.empty()) train_smiles.push_back(line);
    }

    vector<float> train_matrix(train_smiles.size() * DIM, 0.0f);
    size_t valid_train = 0;

    #pragma omp parallel for schedule(dynamic)
    for (long i = 0; i < (long)train_smiles.size(); i++) {

        unique_ptr<ROMol> mol;

        try {
            mol.reset(SmilesToMol(train_smiles[i]));
        } catch (...) {
            #pragma omp critical
            error_out << train_smiles[i] << "\n";
            continue;
        }

        if (!mol) {
            #pragma omp critical
            error_out << train_smiles[i] << "\n";
            continue;
        }

        unique_ptr<ExplicitBitVect> fp(
            MorganFingerprints::getFingerprintAsBitVect(*mol, radius, nbits_fp)
        );

        size_t idx;

        #pragma omp atomic capture
        idx = valid_train++;

        fp_to_float_inplace(fp.get(), train_matrix.data() + idx * DIM, DIM);
    }

    faiss::IndexFlatL2 quantizer(DIM);
    faiss::IndexIVFPQ index(&quantizer, DIM, nlist, m, nbits);

    cout << " Training on " << valid_train << "\n";
    index.train(valid_train, train_matrix.data());

    // ---------- STREAM ADD ----------
    ifstream fin(input_file);
    ofstream offset_out(out_offsets, ios::binary);

    vector<string> batch;
    batch.reserve(batch_size);

    vector<float> matrix(batch_size * DIM);
    vector<char> valid(batch_size);
    vector<uint64_t> offsets(batch_size);

    size_t total_added = 0;

    while (true) {

        streampos pos = fin.tellg();
        if (!getline(fin, line)) break;

        if (line.empty()) continue;

        batch.push_back(line);
        offsets[batch.size() - 1] = (uint64_t)pos;

        if (batch.size() == batch_size) {

            fill(valid.begin(), valid.end(), 0);

            #pragma omp parallel for schedule(dynamic)
            for (long i = 0; i < (long)batch.size(); i++) {

                unique_ptr<ROMol> mol;

                try {
                    mol.reset(SmilesToMol(batch[i]));
                } catch (...) {
                    #pragma omp critical
                    error_out << batch[i] << "\n";
                    continue;
                }

                if (!mol) {
                    #pragma omp critical
                    error_out << batch[i] << "\n";
                    continue;
                }

                unique_ptr<ExplicitBitVect> fp(
                    MorganFingerprints::getFingerprintAsBitVect(*mol, radius, nbits_fp)
                );

                fp_to_float_inplace(fp.get(), matrix.data() + i * DIM, DIM);
                valid[i] = 1;
            }

            size_t ptr = 0;

            for (size_t i = 0; i < batch.size(); i++) {
                if (!valid[i]) continue;

                copy(matrix.begin() + i * DIM,
                     matrix.begin() + (i + 1) * DIM,
                     matrix.begin() + ptr);

                offset_out.write((char*)&offsets[i], sizeof(uint64_t));

                ptr += DIM;
                total_added++;
            }

            if (ptr > 0)
                index.add(ptr / DIM, matrix.data());

            // ---------- ETA ----------
            double progress = (double)total_added / total_lines * 100.0;

            auto now = chrono::high_resolution_clock::now();
            double elapsed = chrono::duration<double>(now - global_start).count();

            double speed = total_added / elapsed;
            double remaining = (total_lines - total_added) / speed;

            cout << fixed << setprecision(2);

            cout << " Added: " << total_added
                 << " | " << progress << "% "
                 << " | " << elapsed / 60 << " min"
                 << " | ETA: " << remaining / 60 << " min"
                 << " |  " << speed << " mol/s"
                 << "\n";

            batch.clear();
        }
    }

    cout << "Writing index...\n";
    faiss::write_index(&index, out_index.c_str());

    cout << " DONE\n";
}