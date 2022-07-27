
atom(org("FBI")).
atom(peop("Doug_Beldon")).
atom(loc("Albuquerque")).
atom(loc("N.M.")).
atom(peop("Henley")).
atom(loc("Washington.")).
atom(orgbasedIn("FBI","N.M.")).
atom(locatedIn("Albuquerque","N.M.")).
atom(liveIn("Doug_Beldon","Albuquerque")).
atom(orgbasedIn("FBI","Albuquerque")).
atom(locatedIn("Doug_Beldon","FBI")).
atom(liveIn("Doug_Beldon","N.M.")).


ok(org("FBI"))
ok(peop("Doug_Beldon"))
ok(loc("Albuquerque"))
ok(loc("N.M."))
ok(peop("Henley"))
ok(loc("Washington."))
ok(orgbasedIn("FBI","N.M."))
ok(locatedIn("Albuquerque","N.M."))
ok(liveIn("Doug_Beldon","Albuquerque"))
ok(orgbasedIn("FBI","Albuquerque"))
ok(liveIn("Doug_Beldon","N.M."))

=>
org("FBI")
locatedIn("Albuquerque","N.M.")
peop("Doug_Beldon")
peop("Henley")
loc("Albuquerque")
loc("N.M.")
loc("Washington.")
orgbasedIn("FBI","N.M.")
orgbasedIn("FBI","Albuquerque")
liveIn("Doug_Beldon","Albuquerque")
liveIn("Doug_Beldon","N.M.")
