atom(org("ANSA")).
atom(loc("Umbria")).
atom(loc("Lazio")).
atom(loc("Italy")).
atom(loc("Campania")).
atom(orgbasedIn("ANSA","Italy")).
atom(locatedIn("ANSA","Campania")).
atom(locatedIn("ANSA","Lazio")).
atom(locatedIn("Campania","Italy")).
atom(locatedIn("Lazio","Italy")).
atom(orgbasedIn("ANSA","Umbria")).
atom(locatedIn("Umbria","Italy")).

% With new regulations, solution time = 88 sec
% Without regulation, solution time = stuck


ok(org("ANSA"))
ok(loc("Umbria"))
ok(loc("Lazio"))
ok(loc("Italy"))
ok(loc("Campania"))
ok(orgbasedIn("ANSA","Italy"))
ok(locatedIn("Campania","Italy"))
ok(locatedIn("Lazio","Italy"))
ok(orgbasedIn("ANSA","Umbria"))
ok(locatedIn("Umbria","Italy"))


