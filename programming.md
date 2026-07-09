# Discover

User: DAU, ingestion/query frequency, session duration

Functional Requirements: UX, feature, session memory, continous improvement

Data

* Numberic: min/max, negative, zero, null, `float("inf")`
* Array: emtpy, null, single element, all same, (un)sorted, (non)decreasing, (non)constant change, outliers at end
* String: ANSII, special chars
 
System: CPU time / memory / I/O operation constraints

Quality: test first?

Trade-off: speed vs memory vs accuracy, batch vs real-time (stream?), CAP

# Design

Brute force (verbal only)

* enumerate/simulate then sort & filter
* identify bottleneck: duplicate/expensive calculation, unnecessary/temporary storage

Optimised (verbal then code)

* Python idiomatic: list comprehension, generative `yield` function, swap operator (`a,b=b,a`)





