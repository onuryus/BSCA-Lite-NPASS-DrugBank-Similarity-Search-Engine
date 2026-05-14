#pragma once
#include <string>
#include <vector>

namespace bscs {

// A target/enzyme/transporter/carrier entry within a DrugBank drug.
struct DrugBankTarget {
    std::string db_id;              // e.g. "BE0000048"
    std::string name;
    std::string organism;
    std::vector<std::string> actions;  // e.g. {"inhibitor"}
    std::string known_action;          // "yes" | "no" | "unknown"

    // From <polypeptide id="P00734" source="Swiss-Prot">
    std::string polypeptide_id;     // UniProt accession
    std::string polypeptide_source; // "Swiss-Prot" | "TrEMBL"
    std::string gene_name;
    std::string general_function;
    std::string specific_function;
    std::string cellular_location;
    std::string chromosome_location;
    std::string locus;
    std::string molecular_weight_pp; // polypeptide molecular weight (string to preserve precision)
    std::string theoretical_pi;
};

// Full record for one DrugBank drug, preserving all XML fields.
struct DrugBankDrug {
    // ── Identifiers ───────────────────────────────────────────────────────
    std::string drugbank_id;          // primary (e.g. "DB00001")
    std::vector<std::string> alt_ids; // secondary / accession IDs
    std::string drug_type;            // "biotech" | "small molecule"
    std::string name;
    std::string cas_number;
    std::string unii;
    std::string state;                // "solid" | "liquid" | "gas"

    // ── Pharmacology text ─────────────────────────────────────────────────
    std::string description;
    std::string indication;
    std::string pharmacodynamics;
    std::string mechanism_of_action;
    std::string toxicity;
    std::string metabolism;
    std::string absorption;
    std::string half_life;
    std::string protein_binding;
    std::string route_of_elimination;
    std::string volume_of_distribution;
    std::string clearance;
    std::string synthesis_reference;

    // ── Classification ────────────────────────────────────────────────────
    std::string classif_direct_parent;
    std::string classif_kingdom;
    std::string classif_superclass;
    std::string classif_class;
    std::string classif_subclass;

    // ── Structure (from <calculated-properties>) ──────────────────────────
    std::string inchikey;
    std::string smiles;
    std::string inchi;
    std::string molecular_formula;
    std::string molecular_weight;
    std::string logp;

    // ── Lists ─────────────────────────────────────────────────────────────
    std::vector<std::string> groups;     // "approved", "withdrawn", etc.
    std::vector<std::string> synonyms;
    std::vector<std::string> categories; // MeSH category names

    // ── Pharmacology actors ───────────────────────────────────────────────
    std::vector<DrugBankTarget> targets;
    std::vector<DrugBankTarget> enzymes;
    std::vector<DrugBankTarget> transporters;
    std::vector<DrugBankTarget> carriers;
};

} // namespace bscs
