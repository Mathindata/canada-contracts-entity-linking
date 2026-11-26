// Q5. A linguistic summary computed inside the graph:
//     "Most vendors with a large total value work with several departments."
// Vendor facts come from the path Contract <- Record -> VendorName -> Vendor;
// a contract counts for every vendor its records resolve to.
// Fuzzy sets (same break points as `python -m cel summarize`):
//   R  large total value: 0 at $total_lo, rising linearly to 1 at $total_hi
//   S  several departments: 1 if at least 2 departments, else 0
//   Q  most: 0 at r = 0.3, rising linearly to 1 at r = 0.8
// r = sum(min(mu_R, mu_S)) / sum(mu_R)   (sigma-counts), truth = mu_most(r)
// Two lines exist only for Kuzu, the engine used in the tests (Neo4j does not
// need them): the collect / UNWIND step restarts the row stream between the two
// aggregations, and sum() is written before count(DISTINCT ...) because the
// other order returned NaN in Kuzu 0.11.
MATCH (c:Contract)<-[:PART_OF]-(:Record)-[:NAMED]->(:VendorName)-[:RESOLVES_TO]->(v:Vendor)
WITH v, c.contract_key AS ck, c.department AS dept, max(c.total_value) AS value
WITH v, sum(value) AS total, count(DISTINCT dept) AS n_departments
WITH collect({total: total, n_departments: n_departments}) AS per_vendor
UNWIND per_vendor AS x
WITH CASE WHEN x.total <= $total_lo THEN 0.0
          WHEN x.total >= $total_hi THEN 1.0
          ELSE (x.total - $total_lo) / ($total_hi - $total_lo) END AS mu_r,
     CASE WHEN x.n_departments >= 2 THEN 1.0 ELSE 0.0 END AS mu_s
WITH count(*) AS nv, sum(mu_r) AS sum_r,
     sum(CASE WHEN mu_r < mu_s THEN mu_r ELSE mu_s END) AS sum_rs
WITH nv, sum_r, sum_rs / sum_r AS r
RETURN nv AS n_vendors, round(sum_r, 2) AS group_size, round(r, 4) AS r,
       round(CASE WHEN r <= 0.3 THEN 0.0 WHEN r >= 0.8 THEN 1.0
                  ELSE (r - 0.3) / 0.5 END, 4) AS truth_most
