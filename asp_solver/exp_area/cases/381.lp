atom(other("June")).
atom(org("State_Department")).
atom(loc("Washington")).
atom(loc("U.S.")).
atom(org("Embassy")).
atom(loc("Ethiopia")).
atom(orgbasedIn("State_Department","U.S.")).
atom(orgbasedIn("Embassy","U.S.")).
atom(orgbasedIn("State_Department","Washington")).
atom(locatedIn("Washington","U.S.")).
atom(locatedIn("U.S.","Ethiopia")).



ok(other("June"))
ok(org("State_Department"))
ok(loc("Washington"))
ok(loc("U.S."))
ok(org("Embassy"))
ok(loc("Ethiopia"))
ok(orgbasedIn("State_Department","U.S."))
ok(orgbasedIn("Embassy","U.S."))
ok(orgbasedIn("State_Department","Washington"))
ok(locatedIn("Washington","U.S."))
ok(locatedIn("U.S.","Ethiopia"))

=>
org("State_Department")
org("Embassy")
locatedIn("Washington","U.S.")
locatedIn("U.S.","Ethiopia")
locatedIn("Washington","Ethiopia")
other("June")
loc("Washington")
loc("U.S.")
loc("Ethiopia")
orgbasedIn("State_Department","U.S.")
orgbasedIn("Embassy","U.S.")
orgbasedIn("State_Department","Washington")
orgbasedIn("State_Department","Ethiopia")
orgbasedIn("Embassy","Ethiopia")