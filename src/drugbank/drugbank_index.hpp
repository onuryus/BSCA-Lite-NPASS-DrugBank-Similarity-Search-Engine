#pragma once
// drugbank_index.hpp — streaming XML parser, binary serialisation, in-RAM index.
//
// Two entry points:
//   load_from_xml(path)   — parse drug_bank.xml (slow, ~10-20 s for full file)
//   load_binary(path)     — load prepare_drugbank output (fast, <1 s)
//
// Lookup:
//   const DrugBankDrug* lookup_by_inchikey(key)  — nullptr if not found

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/tsv_utils.hpp"
#include "drugbank/drugbank_types.hpp"

namespace bscs {

// ── XML tokenizer ─────────────────────────────────────────────────────────────

namespace db_detail {

// Buffered single-char reader with one-char unget.
struct BufR {
    FILE*    f;
    char     buf[1 << 16];
    int      pos = 0, cap = 0, ungot = -1;
    uint64_t consumed = 0;

    explicit BufR(FILE* f_) : f(f_) {}

    int get() {
        if (ungot >= 0) { int c = ungot; ungot = -1; return c; } // already counted
        if (pos == cap) {
            cap = (int)std::fread(buf, 1, sizeof buf, f);
            pos = 0;
            if (!cap) return -1;
        }
        ++consumed;
        return (unsigned char)buf[pos++];
    }
    void unget(int c) { ungot = c; }
};

// Decode XML character references and predefined entities.
// Strips &#13; (CR) entirely so descriptions use LF-only newlines.
static std::string xml_decode(const std::string& s) {
    if (s.find('&') == std::string::npos) return s;
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ) {
        if (s[i] != '&') { out += s[i++]; continue; }
        if      (s.compare(i,5,"&amp;") ==0) { out += '&';  i += 5; }
        else if (s.compare(i,4,"&lt;")  ==0) { out += '<';  i += 4; }
        else if (s.compare(i,4,"&gt;")  ==0) { out += '>';  i += 4; }
        else if (s.compare(i,6,"&quot;")==0) { out += '"';  i += 6; }
        else if (s.compare(i,6,"&apos;")==0) { out += '\''; i += 6; }
        else if (s.compare(i,6,"&#13;") ==0) { /* strip CR */ i += 6; }
        else if (s.compare(i,5,"&#xD;") ==0) { /* strip CR */ i += 5; }
        else {
            size_t sc = s.find(';', i);
            if (sc != std::string::npos) i = sc + 1;
            else out += s[i++];
        }
    }
    return out;
}

// Extract attribute value from a raw attribute string.
static std::string get_attr(const std::string& attrs, const std::string& name) {
    for (char q : {'"', '\''}) {
        std::string pat = name + "=";
        pat += q;
        size_t p = attrs.find(pat);
        if (p != std::string::npos) {
            p += pat.size();
            size_t e = attrs.find(q, p);
            if (e != std::string::npos) return attrs.substr(p, e - p);
        }
    }
    return {};
}

static std::string trim_ws(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && (unsigned char)s[b] <= ' ') ++b;
    while (e > b && (unsigned char)s[e-1] <= ' ') --e;
    return s.substr(b, e - b);
}

// Token types from the XML tokenizer.
enum class TType { OPEN, CLOSE, SELF_CLOSE, TEXT, END_ };
struct Tok {
    TType       type = TType::END_;
    std::string name;
    std::string attrs;
    std::string text;
};

// Parse one XML token from BufR.  Returns false on EOF.
static bool next_tok(BufR& r, Tok& tok) {
    tok = {};

    // Accumulate raw text until '<' (or EOF).
    std::string raw;
    int c;
    while ((c = r.get()) >= 0 && c != '<') raw += (char)c;

    // If we have non-whitespace text, return it and push '<' back.
    if (!raw.empty()) {
        std::string decoded = xml_decode(raw);
        bool all_ws = true;
        for (unsigned char ch : decoded) if (ch > ' ') { all_ws = false; break; }
        if (!all_ws) {
            tok.type = TType::TEXT;
            tok.text = decoded;
            if (c == '<') r.unget(c);
            return true;
        }
        // Whitespace-only — fall through.  c may be '<' or -1.
    }

    if (c < 0) { tok.type = TType::END_; return false; }
    // c == '<'

    int c2 = r.get();
    if (c2 < 0) { tok.type = TType::END_; return false; }

    if (c2 == '/') {
        // CLOSE TAG: read until '>'
        tok.type = TType::CLOSE;
        std::string name_raw;
        while ((c = r.get()) >= 0 && c != '>') name_raw += (char)c;
        tok.name = trim_ws(name_raw);
        return true;
    }

    if (c2 == '?' || c2 == '!') {
        // PI or comment/DOCTYPE — skip to '>'
        while ((c = r.get()) >= 0 && c != '>') {}
        return next_tok(r, tok); // tail-recurse for next real token
    }

    // OPEN / SELF-CLOSE TAG
    r.unget(c2);
    std::string tag;
    bool in_str = false; char sc = 0;
    while ((c = r.get()) >= 0) {
        if (in_str) {
            if (c == sc) in_str = false;
            tag += (char)c;
        } else {
            if (c == '>') break;
            if (c == '"' || c == '\'') { in_str = true; sc = (char)c; tag += (char)c; }
            else tag += (char)c;
        }
    }

    bool self = !tag.empty() && tag.back() == '/';
    if (self) {
        tag.pop_back();
        while (!tag.empty() && (unsigned char)tag.back() <= ' ') tag.pop_back();
    }

    size_t i = 0;
    while (i < tag.size() && (unsigned char)tag[i] > ' ') tok.name += tag[i++];
    while (i < tag.size() && (unsigned char)tag[i] <= ' ') ++i;
    tok.attrs = tag.substr(i);

    tok.type = self ? TType::SELF_CLOSE : TType::OPEN;
    return true;
}

// ── Binary I/O helpers ────────────────────────────────────────────────────────

static void wlist(std::FILE* f, const std::vector<std::string>& v) {
    uint16_t n = (uint16_t)std::min(v.size(), (size_t)65535u);
    std::fwrite(&n, 2, 1, f);
    for (uint16_t i = 0; i < n; ++i) write_str(f, v[i]);
}
static void rlist(std::FILE* f, std::vector<std::string>& v) {
    uint16_t n = 0; std::fread(&n, 2, 1, f);
    v.resize(n);
    for (auto& s : v) s = read_str(f);
}

static void write_target(std::FILE* f, const DrugBankTarget& t) {
    write_str(f, t.db_id);
    write_str(f, t.name);
    write_str(f, t.organism);
    write_str(f, t.known_action);
    write_str(f, t.polypeptide_id);
    write_str(f, t.polypeptide_source);
    write_str(f, t.gene_name);
    write_str(f, t.general_function);
    write_str(f, t.specific_function);
    write_str(f, t.cellular_location);
    write_str(f, t.chromosome_location);
    write_str(f, t.locus);
    write_str(f, t.molecular_weight_pp);
    write_str(f, t.theoretical_pi);
    wlist(f, t.actions);
}
static void read_target(std::FILE* f, DrugBankTarget& t) {
    t.db_id              = read_str(f);
    t.name               = read_str(f);
    t.organism           = read_str(f);
    t.known_action       = read_str(f);
    t.polypeptide_id     = read_str(f);
    t.polypeptide_source = read_str(f);
    t.gene_name          = read_str(f);
    t.general_function   = read_str(f);
    t.specific_function  = read_str(f);
    t.cellular_location  = read_str(f);
    t.chromosome_location= read_str(f);
    t.locus              = read_str(f);
    t.molecular_weight_pp= read_str(f);
    t.theoretical_pi     = read_str(f);
    rlist(f, t.actions);
}
static void wtargets(std::FILE* f, const std::vector<DrugBankTarget>& v) {
    uint16_t n = (uint16_t)std::min(v.size(), (size_t)65535u);
    std::fwrite(&n, 2, 1, f);
    for (uint16_t i = 0; i < n; ++i) write_target(f, v[i]);
}
static void rtargets(std::FILE* f, std::vector<DrugBankTarget>& v) {
    uint16_t n = 0; std::fread(&n, 2, 1, f);
    v.resize(n);
    for (auto& t : v) read_target(f, t);
}

static void write_drug(std::FILE* f, const DrugBankDrug& d) {
    write_str(f, d.drugbank_id);
    write_str(f, d.drug_type);
    write_str(f, d.name);
    write_str(f, d.cas_number);
    write_str(f, d.unii);
    write_str(f, d.state);
    write_str(f, d.description);
    write_str(f, d.indication);
    write_str(f, d.pharmacodynamics);
    write_str(f, d.mechanism_of_action);
    write_str(f, d.toxicity);
    write_str(f, d.metabolism);
    write_str(f, d.absorption);
    write_str(f, d.half_life);
    write_str(f, d.protein_binding);
    write_str(f, d.route_of_elimination);
    write_str(f, d.volume_of_distribution);
    write_str(f, d.clearance);
    write_str(f, d.synthesis_reference);
    write_str(f, d.classif_direct_parent);
    write_str(f, d.classif_kingdom);
    write_str(f, d.classif_superclass);
    write_str(f, d.classif_class);
    write_str(f, d.classif_subclass);
    write_str(f, d.inchikey);
    write_str(f, d.smiles);
    write_str(f, d.inchi);
    write_str(f, d.molecular_formula);
    write_str(f, d.molecular_weight);
    write_str(f, d.logp);
    wlist(f, d.alt_ids);
    wlist(f, d.groups);
    wlist(f, d.synonyms);
    wlist(f, d.categories);
    wtargets(f, d.targets);
    wtargets(f, d.enzymes);
    wtargets(f, d.transporters);
    wtargets(f, d.carriers);
}

static void read_drug(std::FILE* f, DrugBankDrug& d) {
    d.drugbank_id          = read_str(f);
    d.drug_type            = read_str(f);
    d.name                 = read_str(f);
    d.cas_number           = read_str(f);
    d.unii                 = read_str(f);
    d.state                = read_str(f);
    d.description          = read_str(f);
    d.indication           = read_str(f);
    d.pharmacodynamics     = read_str(f);
    d.mechanism_of_action  = read_str(f);
    d.toxicity             = read_str(f);
    d.metabolism           = read_str(f);
    d.absorption           = read_str(f);
    d.half_life            = read_str(f);
    d.protein_binding      = read_str(f);
    d.route_of_elimination = read_str(f);
    d.volume_of_distribution=read_str(f);
    d.clearance            = read_str(f);
    d.synthesis_reference  = read_str(f);
    d.classif_direct_parent= read_str(f);
    d.classif_kingdom      = read_str(f);
    d.classif_superclass   = read_str(f);
    d.classif_class        = read_str(f);
    d.classif_subclass     = read_str(f);
    d.inchikey             = read_str(f);
    d.smiles               = read_str(f);
    d.inchi                = read_str(f);
    d.molecular_formula    = read_str(f);
    d.molecular_weight     = read_str(f);
    d.logp                 = read_str(f);
    rlist(f, d.alt_ids);
    rlist(f, d.groups);
    rlist(f, d.synonyms);
    rlist(f, d.categories);
    rtargets(f, d.targets);
    rtargets(f, d.enzymes);
    rtargets(f, d.transporters);
    rtargets(f, d.carriers);
}

// ── finalize collected text into the right field ──────────────────────────────

static void finalize_field(const std::string& ctx, const std::string& tag,
                            const std::string& raw_buf, const std::string& calc_kind,
                            DrugBankDrug* drug, DrugBankTarget* target) {
    std::string text = xml_decode(trim_ws(raw_buf));
    if (!drug) return;

    if (ctx == "drug_id") {
        if (!text.empty()) drug->drugbank_id = text;
    } else if (ctx == "drug_alt_id") {
        if (!text.empty()) drug->alt_ids.push_back(text);
    } else if (ctx == "group") {
        if (!text.empty()) drug->groups.push_back(text);
    } else if (ctx == "synonym") {
        if (!text.empty()) drug->synonyms.push_back(text);
    } else if (ctx == "category") {
        if (!text.empty()) drug->categories.push_back(text);
    } else if (ctx == "calc_kind") {
        // stored separately; handled by caller
        (void)tag;
    } else if (ctx == "calc_value") {
        if      (calc_kind == "InChIKey")          drug->inchikey          = text;
        else if (calc_kind == "SMILES")            drug->smiles            = text;
        else if (calc_kind == "InChI")             drug->inchi             = text;
        else if (calc_kind == "Molecular Formula") drug->molecular_formula = text;
        else if (calc_kind == "Molecular Weight")  drug->molecular_weight  = text;
        else if (calc_kind == "logP" && drug->logp.empty()) drug->logp     = text;
    } else if (ctx == "classif") {
        if      (tag == "direct-parent") drug->classif_direct_parent = text;
        else if (tag == "kingdom")       drug->classif_kingdom       = text;
        else if (tag == "superclass")    drug->classif_superclass    = text;
        else if (tag == "class")         drug->classif_class         = text;
        else if (tag == "subclass")      drug->classif_subclass      = text;
    } else if (ctx == "drug_field") {
        if      (tag == "name")                   drug->name                   = text;
        else if (tag == "description")            drug->description            = text;
        else if (tag == "cas-number")             drug->cas_number             = text;
        else if (tag == "unii")                   drug->unii                   = text;
        else if (tag == "state")                  drug->state                  = text;
        else if (tag == "indication")             drug->indication             = text;
        else if (tag == "pharmacodynamics")       drug->pharmacodynamics       = text;
        else if (tag == "mechanism-of-action")    drug->mechanism_of_action    = text;
        else if (tag == "toxicity")               drug->toxicity               = text;
        else if (tag == "metabolism")             drug->metabolism             = text;
        else if (tag == "absorption")             drug->absorption             = text;
        else if (tag == "half-life")              drug->half_life              = text;
        else if (tag == "protein-binding")        drug->protein_binding        = text;
        else if (tag == "route-of-elimination")   drug->route_of_elimination   = text;
        else if (tag == "volume-of-distribution") drug->volume_of_distribution = text;
        else if (tag == "clearance")              drug->clearance              = text;
        else if (tag == "synthesis-reference")    drug->synthesis_reference    = text;
    } else if (ctx == "tgt_field" && target) {
        if      (tag == "id")                  target->db_id               = text;
        else if (tag == "name")                target->name                = text;
        else if (tag == "organism")            target->organism            = text;
        else if (tag == "known-action")        target->known_action        = text;
        else if (tag == "gene-name")           target->gene_name           = text;
        else if (tag == "general-function")    target->general_function    = text;
        else if (tag == "specific-function")   target->specific_function   = text;
        else if (tag == "cellular-location")   target->cellular_location   = text;
        else if (tag == "chromosome-location") target->chromosome_location = text;
        else if (tag == "locus")               target->locus               = text;
        else if (tag == "molecular-weight")    target->molecular_weight_pp = text;
        else if (tag == "theoretical-pi")      target->theoretical_pi      = text;
    } else if (ctx == "action" && target) {
        if (!text.empty()) target->actions.push_back(text);
    }
}

} // namespace db_detail

// ── DrugBankIndex ─────────────────────────────────────────────────────────────

class DrugBankIndex {
public:
    // Build a secondary index by canonical SMILES for drugs that have a SMILES
    // but no InChIKey (common for biotech/protein drugs).
    // canon_fn(smiles) → canonical SMILES string (empty on failure).
    void build_smiles_index(std::function<std::string(const std::string&)> canon_fn) {
        by_canon_smiles_.clear();
        by_canon_smiles_.reserve(drugs_.size() * 2);
        for (size_t i = 0; i < drugs_.size(); ++i) {
            if (drugs_[i].smiles.empty()) continue;
            std::string canon = canon_fn(drugs_[i].smiles);
            if (!canon.empty()) by_canon_smiles_.emplace(std::move(canon), i);
        }
    }

    const DrugBankDrug* lookup_by_canon_smiles(const std::string& canon) const {
        if (canon.empty()) return nullptr;
        auto it = by_canon_smiles_.find(canon);
        if (it == by_canon_smiles_.end()) return nullptr;
        return &drugs_[it->second];
    }
    // Load from raw DrugBank XML.
    // progress_cb(bytes_done, total_bytes, drugs_done) called ~every 1000 drugs.
    void load_from_xml(const std::string& path,
                       std::function<void(uint64_t,uint64_t,size_t)> progress_cb = nullptr) {
        using namespace db_detail;

        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("Cannot open DrugBank XML: " + path);

        // Determine file size for progress.
        uint64_t file_size = 0;
        if (std::fseek(f, 0, SEEK_END) == 0) {
            file_size = (uint64_t)std::ftell(f);
            std::fseek(f, 0, SEEK_SET);
        }

        BufR r(f);
        Tok tok;

        // ── parser state ──────────────────────────────────────────────────
        int         depth       = 0;
        int         drug_depth  = -1;
        std::string drug_section;          // current direct child name
        std::string target_section;        // "targets","enzymes","transporters","carriers"
        int         target_depth = -1;
        bool        in_polypeptide = false;
        int         poly_depth   = -1;
        bool        in_calc_props= false;
        bool        in_cat_item  = false;  // inside <categories><category> container
        std::string calc_kind;

        // Collecting state
        bool        coll      = false;
        int         coll_nest = 0;
        std::string coll_ctx, coll_tag, coll_buf;

        // Current drug / target pointers into drugs_
        DrugBankDrug*   cur_drug   = nullptr;
        DrugBankTarget* cur_target = nullptr;

        auto start_coll = [&](const std::string& ctx, const std::string& tag) {
            coll = true; coll_nest = 0; coll_ctx = ctx; coll_tag = tag; coll_buf.clear();
        };

        auto get_target_vec = [&]() -> std::vector<DrugBankTarget>* {
            if (!cur_drug) return nullptr;
            if (target_section == "targets")      return &cur_drug->targets;
            if (target_section == "enzymes")      return &cur_drug->enzymes;
            if (target_section == "transporters") return &cur_drug->transporters;
            if (target_section == "carriers")     return &cur_drug->carriers;
            return nullptr;
        };

        while (next_tok(r, tok)) {

            // ── TEXT ──────────────────────────────────────────────────────
            if (tok.type == TType::TEXT) {
                if (coll) coll_buf += tok.text;
                continue;
            }

            // ── OPEN / SELF-CLOSE ─────────────────────────────────────────
            if (tok.type == TType::OPEN || tok.type == TType::SELF_CLOSE) {
                if (tok.type == TType::OPEN) ++depth;

                if (coll) {
                    // Nested element inside collecting context
                    if (tok.type == TType::OPEN) ++coll_nest;
                    continue;
                }

                const auto& n = tok.name;

                // Root <drug> element
                if (n == "drug" && depth == 2) {
                    drugs_.emplace_back();
                    cur_drug   = &drugs_.back();
                    cur_target = nullptr;
                    cur_drug->drug_type = get_attr(tok.attrs, "type");
                    drug_depth = depth;
                    drug_section.clear();
                    target_section.clear();
                    in_calc_props = false;
                    in_cat_item   = false;
                    calc_kind.clear();
                    continue;
                }

                if (!cur_drug || drug_depth < 0) continue;

                int rel = depth - drug_depth;

                // ── rel == 1: direct children of <drug> ──────────────────
                if (rel == 1) {
                    drug_section = n;
                    in_calc_props = (n == "calculated-properties");
                    if (n == "targets" || n == "enzymes" ||
                        n == "transporters" || n == "carriers") {
                        target_section = n;
                    }

                    if (n == "drugbank-id") {
                        bool primary = (get_attr(tok.attrs, "primary") == "true");
                        start_coll(primary ? "drug_id" : "drug_alt_id", n);
                    } else if (n == "name" || n == "description" || n == "cas-number" ||
                               n == "unii" || n == "state" || n == "indication" ||
                               n == "pharmacodynamics" || n == "mechanism-of-action" ||
                               n == "toxicity" || n == "metabolism" || n == "absorption" ||
                               n == "half-life" || n == "protein-binding" ||
                               n == "route-of-elimination" || n == "volume-of-distribution" ||
                               n == "clearance" || n == "synthesis-reference") {
                        start_coll("drug_field", n);
                    }
                    continue;
                }

                // ── rel == 2: grandchildren ───────────────────────────────
                if (rel == 2) {
                    if (drug_section == "groups" && n == "group") {
                        start_coll("group", n);
                    } else if (drug_section == "synonyms" && n == "synonym") {
                        start_coll("synonym", n);
                    } else if (drug_section == "categories" && n == "category") {
                        in_cat_item = true;
                    } else if (in_calc_props && n == "property") {
                        calc_kind.clear();
                    } else if (drug_section == "classification") {
                        start_coll("classif", n);
                    } else if (!target_section.empty() &&
                               (n == "target" || n == "enzyme" ||
                                n == "transporter" || n == "carrier")) {
                        auto* vec = get_target_vec();
                        if (vec) {
                            vec->emplace_back();
                            cur_target = &vec->back();
                            target_depth = depth;
                            in_polypeptide = false;
                        }
                    }
                    continue;
                }

                // ── rel == 3: great-grandchildren ─────────────────────────
                if (rel == 3) {
                    if (in_cat_item && n == "category") {
                        start_coll("category", n);
                    } else if (in_calc_props && n == "kind") {
                        start_coll("calc_kind", n);
                    } else if (in_calc_props && n == "value") {
                        start_coll("calc_value", n);
                    } else if (cur_target && target_depth == depth - 1) {
                        // direct child of target element
                        if (n == "polypeptide") {
                            cur_target->polypeptide_id     = get_attr(tok.attrs, "id");
                            cur_target->polypeptide_source = get_attr(tok.attrs, "source");
                            in_polypeptide = true;
                            poly_depth = depth;
                        } else if (n != "actions" && n != "references") {
                            start_coll("tgt_field", n);
                        }
                    }
                    continue;
                }

                // ── rel >= 4: deeper nesting ──────────────────────────────
                if (rel >= 4) {
                    if (cur_target) {
                        int tgt_rel = depth - target_depth;
                        if (tgt_rel == 2 && n == "action") {
                            start_coll("action", n);
                        } else if (in_polypeptide && poly_depth >= 0 &&
                                   depth == poly_depth + 1 &&
                                   n != "external-identifiers" && n != "synonyms") {
                            start_coll("tgt_field", n);
                        }
                    }
                }
                continue;
            }

            // ── CLOSE ─────────────────────────────────────────────────────
            if (tok.type == TType::CLOSE) {
                if (coll) {
                    if (coll_nest > 0) {
                        --coll_nest;
                    } else {
                        // Finalize collection
                        std::string kind_out;
                        if (coll_ctx == "calc_kind") {
                            calc_kind = db_detail::xml_decode(db_detail::trim_ws(coll_buf));
                        } else {
                            finalize_field(coll_ctx, coll_tag, coll_buf, calc_kind,
                                           cur_drug, cur_target);
                        }
                        coll = false;
                        coll_buf.clear();
                    }
                    --depth;
                    continue;
                }

                const auto& n = tok.name;

                if (n == "drug" && depth == drug_depth) {
                    cur_drug   = nullptr;
                    cur_target = nullptr;
                    drug_depth = -1;
                    if (progress_cb && (drugs_.size() % 500 == 0))
                        progress_cb(r.consumed, file_size, drugs_.size());
                } else if (!target_section.empty() &&
                           n == target_section && depth == drug_depth + 1) {
                    target_section.clear();
                    cur_target = nullptr;
                    target_depth = -1;
                } else if (cur_target && depth == target_depth) {
                    cur_target   = nullptr;
                    target_depth = -1;
                    in_polypeptide = false;
                    poly_depth   = -1;
                } else if (n == "calculated-properties") {
                    in_calc_props = false;
                    calc_kind.clear();
                } else if (n == "category" && in_cat_item && depth == drug_depth + 2) {
                    in_cat_item = false;
                } else if (n == "polypeptide" && in_polypeptide) {
                    in_polypeptide = false;
                    poly_depth = -1;
                } else if (n == drug_section && depth == drug_depth + 1) {
                    drug_section.clear();
                }

                --depth;
                continue;
            }
        } // while next_tok

        std::fclose(f);

        if (progress_cb)
            progress_cb(file_size, file_size, drugs_.size());

        build_index();
    }

    // ── Binary save / load ────────────────────────────────────────────────────
    static constexpr uint32_t MAGIC = 0x44424958u; // 'DBIX'

    void save_binary(const std::string& path) const {
        std::FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) throw std::runtime_error("Cannot write: " + path);
        std::fwrite(&MAGIC, 4, 1, f);
        uint32_t n = (uint32_t)drugs_.size();
        std::fwrite(&n, 4, 1, f);
        for (const auto& d : drugs_) db_detail::write_drug(f, d);
        std::fclose(f);
    }

    void load_binary(const std::string& path) {
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("Cannot open binary DrugBank index: " + path);
        uint32_t magic = 0, n = 0;
        std::fread(&magic, 4, 1, f);
        if (magic != MAGIC) { std::fclose(f); throw std::runtime_error("Bad magic in " + path); }
        std::fread(&n, 4, 1, f);
        drugs_.resize(n);
        for (auto& d : drugs_) db_detail::read_drug(f, d);
        std::fclose(f);
        build_index();
    }

    // ── Lookup ────────────────────────────────────────────────────────────────
    const DrugBankDrug* lookup_by_inchikey(const std::string& ik) const {
        auto it = by_inchikey_.find(ik);
        if (it == by_inchikey_.end()) return nullptr;
        return &drugs_[it->second];
    }

    size_t drug_count() const { return drugs_.size(); }

    const std::vector<DrugBankDrug>& drugs() const { return drugs_; }

private:
    std::vector<DrugBankDrug>              drugs_;
    std::unordered_map<std::string,size_t> by_inchikey_;
    std::unordered_map<std::string,size_t> by_canon_smiles_;

    void build_index() {
        by_inchikey_.clear();
        by_inchikey_.reserve(drugs_.size() * 2);
        for (size_t i = 0; i < drugs_.size(); ++i) {
            if (!drugs_[i].inchikey.empty())
                by_inchikey_.emplace(drugs_[i].inchikey, i);
        }
    }
};

} // namespace bscs
