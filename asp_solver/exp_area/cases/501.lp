
atom(loc("Pittsburgh")).
atom(loc("Dade_County")).
atom(loc("Fla.")).
atom(loc("Rochester")).
atom(loc("N.Y.")).
atom(locatedIn("Dade_County","Fla.")).
atom(locatedIn("Dade_County","Pittsburgh")).
atom(locatedIn("Dade_County","N.Y.")).
atom(locatedIn("Pittsburgh","Fla.")).
atom(locatedIn("Rochester","N.Y.")).
atom(locatedIn("Pittsburgh","Dade_County")).

% Before directional rules
ok(loc("Pittsburgh"))
ok(loc("Dade_County"))
ok(loc("Fla."))
ok(loc("Rochester"))
ok(loc("N.Y."))
ok(locatedIn("Dade_County","Fla."))
ok(locatedIn("Dade_County","Pittsburgh"))
ok(locatedIn("Dade_County","N.Y."))
ok(locatedIn("Pittsburgh","Fla."))
ok(locatedIn("Rochester","N.Y."))
ok(locatedIn("Pittsburgh","Dade_County"))

=> ok(locatedIn("Dade_County","Pittsburgh")) & ok(locatedIn("Pittsburgh","Dade_County")): contradiction


% After directional rules
ok(loc("Pittsburgh"))
ok(loc("Dade_County"))
ok(loc("Fla."))
ok(loc("Rochester"))
ok(loc("N.Y."))
ok(locatedIn("Dade_County","Fla."))
ok(locatedIn("Dade_County","Pittsburgh"))
ok(locatedIn("Dade_County","N.Y."))
ok(locatedIn("Pittsburgh","Fla."))
ok(locatedIn("Rochester","N.Y."))

=>
locatedIn("Dade_County","Fla.")
locatedIn("Dade_County","Pittsburgh")
locatedIn("Dade_County","N.Y.")
locatedIn("Pittsburgh","Fla.")
locatedIn("Rochester","N.Y.")
loc("Pittsburgh")
loc("Dade_County")
loc("Fla.")
loc("Rochester")
loc("N.Y.")


=====================
ok(loc("Pittsburgh"))
ok(loc("Dade_County"))
ok(loc("Fla."))
ok(loc("Rochester"))
ok(loc("N.Y."))
ok(locatedIn("Dade_County","Fla."))
ok(locatedIn("Dade_County","N.Y."))
ok(locatedIn("Pittsburgh","Fla."))
ok(locatedIn("Rochester","N.Y."))
ok(locatedIn("Pittsburgh","Dade_County"))

=>
locatedIn("Dade_County","Fla.")
locatedIn("Dade_County","N.Y.")
locatedIn("Pittsburgh","Fla.")
locatedIn("Rochester","N.Y.")
locatedIn("Pittsburgh","Dade_County")
locatedIn("Pittsburgh","N.Y.")
loc("Pittsburgh")
loc("Dade_County")
loc("Fla.")
loc("Rochester")
loc("N.Y.")


Intersection
loc("Pittsburgh")
loc("Dade_County")
loc("Fla.")
loc("Rochester")
loc("N.Y.")
locatedIn("Dade_County","Fla.")
locatedIn("Dade_County","N.Y.")
locatedIn("Pittsburgh","Fla.")
locatedIn("Rochester","N.Y.")

Contradiction
locatedIn("Pittsburgh","N.Y.")
locatedIn("Pittsburgh","Fla.")
=> If location A is located in location B and A is also located in C
=> B must located in C or C must located in B


Not only intersection but contradiction between answer sets


