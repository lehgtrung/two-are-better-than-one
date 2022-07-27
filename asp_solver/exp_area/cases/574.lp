
atom(loc("United_States")).
atom(loc("Britain")).
atom(loc("France")).
atom(org("Security_Council")).
atom(loc("U.S.")).
atom(org("Navy")).
atom(other("Jan.")).
atom(orgbasedIn("Security_Council","U.S.")).
atom(orgbasedIn("Navy","U.S.")).
atom(locatedIn("U.S.","United_States")).
atom(locatedIn("Security_Council","France")).


=>
ok(loc("United_States"))
ok(loc("Britain"))
ok(loc("France"))
ok(org("Security_Council"))
ok(loc("U.S."))
ok(org("Navy"))
ok(other("Jan."))
ok(orgbasedIn("Security_Council","U.S."))
ok(orgbasedIn("Navy","U.S."))
ok(locatedIn("U.S.","United_States"))