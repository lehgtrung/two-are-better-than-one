% Type inference

loc(Q) :- locatedIn(P,Q)
loc(Q) :- locatedIn(P,Q)

org(X) :- orgbasedIn(O,P)
loc(P) :- orgbasedIn(O,P)

peop(X) :- liveIn(X,P)
loc(P) :- liveIn(X,P)

peop(X) :- workFor(X,O)
org(O) :- workFor(X,O)

peop(X) :- kill(X,Y)
peop(Y) :- kill(X,Y)

% Triplets
orgbasedIn(O, P2) :- locatedIn(P1,P2), orgbasedIn(O,P1).
locatedIn(P1, P3) :- locatedIn(P1,P2), locatedIn(P2,P3).
liveIn(X,P2) :- liveIn(X,P1), locatedIn(P1,P2).

% Not absolutely correct, we assume no one works remotely
liveIn(X,P) :- workFor(X,O), orgbasedIn(O,P).
orgbasedIn(O,P) :- liveIn(X,P), workFor(X,O).
