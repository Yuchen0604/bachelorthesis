import json
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

LAGRANGE_PATH = os.path.join(base_dir, "relations_lagrange", "220_lagrange_relations.json")
REBEL_ALL     = os.path.join(base_dir, "relations_rebel", "nonorig_relations.json")

#IDs for the 34 Lagrange-only relations (not in REBEL at all)
manual_ids = {
    "UTC date of spacecraft launch":           "P619",
    "area":                                    "P2046",
    "birth name":                              "P1477",
    "date of birth":                           "P569",
    "date of death":                           "P570",
    "date of first performance":               "P1191",
    "date of official opening":                "P1619",
    "demonym":                                 "P1549",
    "dissolved, abolished or demolished date": "P576",
    "elevation above sea level":               "P2044",
    "end time":                                "P582",
    "female form of label":                    "P2521",
    "has part or parts":                       "P527",
    "has use":                                 "P366",
    "inception":                               "P571",
    "length":                                  "P2043",
    "located in/on physical feature":          "P706",
    "made from material":                      "P186",
    "male form of label":                      "P3321",
    "name":                                    "P2561",
    "name in native language":                 "P1559",
    "nickname":                                "P1449",
    "official name":                           "P1448",
    "point in time":                           "P585",
    "population":                              "P1082",
    "publication date":                        "P577",
    "short name":                              "P1813",
    "start time":                              "P580",
    "street address":                          "P6375",
    "taxon common name":                       "P1843",
    "time of discovery or invention":          "P575",
    "title":                                   "P1476",
    "uses capitalization for":                 "P6106",
    "work period (start)":                     "P2031",
}


def main():
    with open(LAGRANGE_PATH, encoding="utf-8") as f:
        lagrange = json.load(f)
    with open(REBEL_ALL, encoding="utf-8") as f:
        rebel_all = json.load(f)

    rebel_label_to_id = {
        e["predicate_label"]: e["predicate_id"]
        for e in rebel_all
        if e.get("predicate_id")
    }

    #fill predicate_ids from REBEL where possible
    missing = []
    for r in lagrange:
        pid = rebel_label_to_id.get(r["predicate_label"], "")
        r["predicate_id"] = pid
        if not pid:
            missing.append(r)

    print(f"Filled {len(lagrange) - len(missing)}/{len(lagrange)} IDs from REBEL.")

    #fill remaining from hardcoded list
    not_found = []
    for r in missing:
        pid = manual_ids.get(r["predicate_label"], "")
        r["predicate_id"] = pid
        if not pid:
            not_found.append(r["predicate_label"])

    print(f"Filled {len(missing) - len(not_found)}/{len(missing)} IDs from hardcoded list.")

    if not_found:
        print(f"WARNING: no ID found for: {not_found}")

    with open(LAGRANGE_PATH, "w", encoding="utf-8") as f:
        json.dump(lagrange, f, ensure_ascii=False, indent=2)
    print(f"\nSaved predicate IDs to {LAGRANGE_PATH}")


if __name__ == "__main__":
    main()
