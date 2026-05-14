# BSCA Lite
Natural Product & Drug Similarity Search

BSCA Lite is a fast chemical similarity search tool built with RDKit and C++20. It searches query molecules against **NPASS** (natural products) and **DrugBank** (approved/investigational drugs) databases using Morgan fingerprints and Tanimoto similarity.

No large commercial databases or GPU required — just NPASS and DrugBank data, which are freely available.

---

## Features

- Morgan fingerprint similarity search (Tanimoto, Dice, Tversky, Cosine)
- Pre-computed fingerprint index for fast NPASS search (~50 ms per query)
- Full NPASS annotation: biological activities, toxicity, species, targets
- Full DrugBank annotation: pharmacology, mechanism, targets, enzymes
- Interactive multi-query session mode
- TSV output (per-database files)

---

## Requirements

- Linux x86-64
- GCC 13+
- CMake ≥ 3.16
- Conda / Mamba

---

## Installation

### 1. Create the Conda environment

```bash
mamba create -n chem_cpp -c conda-forge \
    cmake make gcc gxx \
    boost-cpp eigen \
    faiss-gpu \
    -y

conda activate chem_cpp
```

### 2. Build and install RDKit

```bash
git clone https://github.com/rdkit/rdkit.git
cd rdkit && mkdir build && cd build

cmake \
    -DRDK_BUILD_PYTHON_WRAPPERS=OFF \
    -DRDK_INSTALL_INTREE=OFF \
    -DRDK_BUILD_FREETYPE_SUPPORT=OFF \
    -DRDK_BUILD_CAIRO_SUPPORT=OFF \
    -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
    ..

make -j$(nproc) && make install
cd ../..
```

### 3. Clone and build BSCA Lite

```bash
git clone https://github.com/onuryus/BSCA-Lite-NPASS-DrugBank-Similarity-Search-Engine.git
cd BSCA-Lite-NPASS-DrugBank-Similarity-Search-Engine

mkdir build_dir && cd build_dir
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) search_lite prepare_npass prepare_drugbank
cp search_lite prepare_npass prepare_drugbank ../build/
cd ..
```

---

## Configuration

```bash
cp build/config.sh.example build/config.sh
nano build/config.sh
```

Fill in your paths:

```bash
NPASS_RAW="/path/to/npass"          # directory with NPASS TSV files
DRUGBANK_XML="/path/to/drugbank.xml" # DrugBank full database XML
IDIR="/path/to/index"               # output directory for binary indexes
```

---

## Data Sources

| Database | Access | Size |
|---|---|---|
| **NPASS** | [bidd.group/NPASS](https://bidd.group/NPASS/) — free download | ~200K natural products |
| **DrugBank** | [go.drugbank.com](https://go.drugbank.com/releases/latest) — free academic account | ~20K drugs |

Download the full NPASS dataset (all TSV files) and the DrugBank full database XML export.

---

## Building the Index

Run once before the first search session (~2–3 minutes):

```bash
conda activate chem_cpp
cd BSCA-lite
bash build/prepare_databases.sh
```

This builds:
- NPASS binary lookup index
- NPASS pre-computed fingerprints (`npass_fps.bin`) for fast search
- DrugBank binary index

---

## Searching

```bash
conda activate chem_cpp
cd BSCA-lite
bash build/search_lite.sh
```

The program loads databases once and runs an interactive session:

```
Select annotation databases:
  [2] NPASS only
  [3] DrugBank only
  [6] NPASS + DrugBank    ← default

Choice [default=6]:

Select output format:
  [1] combined
  [2] per-db
  [3] both

Enter query SMILES (or 'quit'): CC(=O)Oc1ccccc1C(=O)O

[PHASE] Phase 2: NPASS similarity search
  [OK]  50 hits  |  top: 1.0000  |  16 ms

[PHASE] Phase 3: DrugBank similarity search
  [OK]  50 hits  |  top: 1.0000  |  108 ms
```

Results are written to `$IDIR/results_npass.tsv` and `$IDIR/results_drugbank.tsv`.

---

## Output Format

### results_npass.tsv

| Column | Description |
|---|---|
| rank, score, metric | Similarity rank and Tanimoto score |
| np_id, smiles_npass | NPASS compound ID and SMILES |
| inchi, inchikey | Structure identifiers |
| generalinfo columns | Molecular formula, MW, logP, etc. |
| n_activities, activities_detail | Biological activities with targets |
| n_toxicities, toxicities_detail | Toxicity records |
| n_species, species_detail | Organism and species pairs |

### results_drugbank.tsv

| Column | Description |
|---|---|
| rank, score, metric | Similarity rank and Tanimoto score |
| drugbank_id, name | DrugBank identifier and drug name |
| groups, categories | Drug groups and ATC categories |
| description, indication | Clinical description and indication |
| mechanism_of_action | Mechanism of action |
| toxicity_text, metabolism | Toxicity and metabolic data |
| n_targets, targets_detail | Protein targets with UniProt IDs |
| n_enzymes, enzymes_detail | Metabolizing enzymes |

---

## Performance

Benchmarked on a CPU (12 threads):

| Phase | Time |
|---|---|
| Database load (first query only) | ~2 s |
| NPASS search (203K compounds, pre-computed FPs) | ~20 ms |
| DrugBank search (20K compounds) | ~120 ms |
| **Total per query** | **~0.2 s** |

---

## Search Parameters

| Flag | Default | Description |
|---|---|---|
| `--metric` | `tanimoto` | Similarity metric: `tanimoto`, `dice`, `tversky`, `cosine` |
| `--top` | `50` | Number of hits written to output |
| `--databases` | — | `npass`, `drugbank`, or `npass,drugbank` |
| `--out-mode` | — | `combined`, `per-db`, or `both` |
| `--query` | — | Single non-interactive query (optional) |
