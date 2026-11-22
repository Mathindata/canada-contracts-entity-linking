// Q3. Provenance: every raw spelling behind one vendor, with its link
// probability and example source rows. Pass $vendor_id, or leave it empty to
// see the vendor with the most spellings.
MATCH (v:Vendor)
WHERE ($vendor_id <> '' AND v.entity_id = $vendor_id) OR ($vendor_id = '' AND v.n_names > 1)
WITH v ORDER BY v.n_names DESC, v.entity_id LIMIT 1
MATCH (n:VendorName)-[l:RESOLVES_TO]->(v)
OPTIONAL MATCH (r:Record)-[:NAMED]->(n)
WITH v, n, l, collect(r.reference_number) AS rows
RETURN v.entity_id AS vendor_id, v.canonical_name AS vendor, n.vendor_name AS raw_name,
       n.fsa AS postal_prefix, round(l.prob, 3) AS link_probability, size(rows) AS n_rows,
       rows[0..5] AS example_rows
ORDER BY link_probability, raw_name
