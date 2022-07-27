atom(peop("Elmore")).
atom(other("Dutch")).
atom(peop("Leonard")).
atom(loc("Birmingham")).
atom(peop("Tom_Selleck")).
atom(loc("Detroit")).
atom(loc("California.")).
atom(locatedIn("Tom_Selleck","Birmingham")).
atom(liveIn("Elmore","Birmingham")).
atom(liveIn("Tom_Selleck","Detroit")).
atom(liveIn("Leonard","Birmingham")).



ok(peop("Elmore")),
ok(other("Dutch")),
ok(peop("Leonard")),
ok(loc("Birmingham")),
ok(peop("Tom_Selleck")),
ok(loc("Detroit")),
ok(loc("California.")),
ok(liveIn("Elmore","Birmingham")),
ok(liveIn("Tom_Selleck","Detroit")),
ok(liveIn("Leonard","Birmingham"))


ok(peop("Elmore")),
ok(other("Dutch")),
ok(peop("Leonard")),
ok(loc("Birmingham")),
ok(loc("Detroit")),
ok(loc("California.")),
ok(locatedIn("Tom_Selleck","Birmingham")),
ok(liveIn("Elmore","Birmingham")),
ok(liveIn("Tom_Selleck","Detroit")),
ok(liveIn("Leonard","Birmingham"))