// Q2. Early-warning signal: contracts that grew through amendments.
// growth = total value / original value. Each row names the vendor, the
// weakest vendor link behind it, and the source rows to check.
MATCH (d:Department)-[:AWARDED]->(c:Contract)
WHERE c.n_amendments >= $min_amendments OR coalesce(c.growth, 0.0) >= $min_growth
MATCH (c)<-[:PART_OF]-(r:Record)-[:NAMED]->(:VendorName)-[l:RESOLVES_TO]->(v:Vendor)
WITH d, c, v, min(l.prob) AS link_conf, collect(r.reference_number) AS refs
RETURN d.name AS department, c.procurement_id AS procurement_id, v.canonical_name AS vendor,
       c.n_amendments AS amendments, round(c.original_value, 2) AS original_value,
       round(c.total_value, 2) AS total_value, round(coalesce(c.growth, 0.0), 2) AS growth,
       c.end_date AS end_date, round(link_conf, 3) AS vendor_link_confidence,
       refs AS source_rows
ORDER BY growth DESC, amendments DESC, procurement_id
LIMIT 50
