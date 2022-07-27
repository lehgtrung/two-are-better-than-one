
atom(other("11_million")).
atom(other("2.5_million")).
atom(peop("Richard_Kahn")).
atom(org("American_Diabetes_Association")).
atom(loc("Alexandria")).
atom(loc("Va.")).
atom(workFor("Richard_Kahn","American_Diabetes_Association")).
atom(liveIn("Richard_Kahn","Va.")).
atom(orgbasedIn("American_Diabetes_Association","Va.")).
atom(orgbasedIn("American_Diabetes_Association","Alexandria")).
atom(locatedIn("Alexandria","Va.")).


ok(other("11_million"))
ok(other("2.5_million"))
ok(peop("Richard_Kahn"))
ok(org("American_Diabetes_Association"))
ok(loc("Alexandria"))
ok(loc("Va."))
ok(workFor("Richard_Kahn","American_Diabetes_Association"))
ok(liveIn("Richard_Kahn","Va."))
ok(orgbasedIn("American_Diabetes_Association","Va."))
ok(orgbasedIn("American_Diabetes_Association","Alexandria"))
ok(locatedIn("Alexandria","Va."))

=>
org("American_Diabetes_Association")
locatedIn("Alexandria","Va.")
peop("Richard_Kahn")
other("11_million")
other("2.5_million")
loc("Alexandria")
loc("Va.")
orgbasedIn("American_Diabetes_Association","Va.")
orgbasedIn("American_Diabetes_Association","Alexandria")
liveIn("Richard_Kahn","Va.")
liveIn("Richard_Kahn","Alexandria")
workFor("Richard_Kahn","American_Diabetes_Association")