// Q1. Multi-hop: vendors that hold contracts with two different departments.
// Path: Department -> Contract <- Record -> VendorName -> Vendor.
// The only uncertain hop is RESOLVES_TO. The claim "vendor V works with A and B"
// joins two such paths, so its confidence is the product of the best link
// probability on each side (independence assumed). Results keep source rows.
MATCH (d:Department)-[:AWARDED]->(c:Contract)<-[:PART_OF]-(r:Record)-[:NAMED]->(:VendorName)-[l:RESOLVES_TO]->(v:Vendor)
WITH v, d, max(l.prob) AS conf, count(DISTINCT c) AS n_contracts,
     collect(DISTINCT r.reference_number) AS refs
WITH v, collect({dept: d.name, conf: conf, n_contracts: n_contracts, refs: refs[0..3]}) AS per_dept
WHERE size(per_dept) >= 2
UNWIND per_dept AS a
UNWIND per_dept AS b
WITH v, a, b
WHERE a.dept < b.dept AND a.conf * b.conf >= $min_conf
RETURN v.entity_id AS vendor_id, v.canonical_name AS vendor,
       a.dept AS department_1, a.n_contracts AS contracts_1,
       b.dept AS department_2, b.n_contracts AS contracts_2,
       round(a.conf * b.conf, 3) AS path_confidence,
       a.refs AS example_rows_1, b.refs AS example_rows_2
ORDER BY contracts_1 + contracts_2 DESC, path_confidence DESC, vendor_id
LIMIT 50
