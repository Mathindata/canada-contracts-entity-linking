// Q4. Review queue: accepted links the model is least sure about
// ($low <= P(match) < $high). These are the pairs a person should check first.
MATCH (a:VendorName)-[s:SAME_AS]->(b:VendorName)
WHERE s.prob >= $low AND s.prob < $high
RETURN a.vendor_name AS name_a, a.fsa AS postal_a, b.vendor_name AS name_b,
       b.fsa AS postal_b, round(s.prob, 3) AS probability
ORDER BY probability, name_a
LIMIT 50
