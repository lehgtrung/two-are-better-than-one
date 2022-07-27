atom(loc("STOCKTON-ON-TEES")).
atom(loc("England")).
atom(org("AP")).
atom(peop("Michael_Minns")).
atom(other("$160_,_000")).
atom(other("World_War_II")).
atom(orgbasedIn("AP","England")).
atom(orgbasedIn("AP","STOCKTON-ON-TEES")).
atom(locatedIn("Michael_Minns","STOCKTON-ON-TEES")).
atom(locatedIn("STOCKTON-ON-TEES","England")).
atom(liveIn("Michael_Minns","England")).


% 2 answer sets

% ok(loc("STOCKTON-ON-TEES")),
% ok(loc("England")),
% ok(org("AP")),
% ok(peop("Michael_Minns")),
% ok(other("$160_,_000")),
% ok(other("World_War_II")),
% ok(orgbasedIn("AP","England")),
% ok(orgbasedIn("AP","STOCKTON-ON-TEES")),
% ok(locatedIn("STOCKTON-ON-TEES","England")),
% ok(liveIn("Michael_Minns","England"))

% =================================================
% ok(loc("STOCKTON-ON-TEES")),
% ok(loc("England")),
% ok(org("AP")),
% ok(other("$160_,_000")),
% ok(other("World_War_II")),
% ok(orgbasedIn("AP","England")),
% ok(orgbasedIn("AP","STOCKTON-ON-TEES")),
% ok(locatedIn("Michael_Minns","STOCKTON-ON-TEES")),
% ok(locatedIn("STOCKTON-ON-TEES","England")),
% ok(liveIn("Michael_Minns","England"))
% => loc("Michael_Minns") => UNSATISFIABLE
