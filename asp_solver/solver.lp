
% Type requirements
% peop(X), loc(P) :- liveIn(X,P).
% loc(P), loc(Q) :- locatedIn(P,Q).
% org(O), loc(P) :- orgbasedIn(O,P).
% peop(X), org(O) :- workFor(X,O).
% peop(X), peop(Y):- kill(X,Y).

peop(X) :- liveIn(X,P).
loc(P) :- locatedIn(P,Q).
org(O) :- orgbasedIn(O,P).
peop(X) :- workFor(X,O).
peop(X) :- kill(X,Y).

loc(P) :- liveIn(X,P).
loc(Q) :- locatedIn(P,Q).
loc(P) :- orgbasedIn(O,P).
org(O) :- workFor(X,O).
peop(Y):- kill(X,Y).


% Relation symmetricity
% {liveIn(X,P); liveIn(P,X)}.
% {locatedIn(P,Q); locatedIn(Q,P)}.
% {orgbasedIn(O,P); orgbasedIn(P,O)}.
% {workFor(X,O); workFor(O,X)}.
% {kill(X,Y); kill(Y,X)}.


% Other requirements
:- liveIn(X,X).
:- locatedIn(X,X).
:- orgbasedIn(X,X).
:- workFor(X,X).
:- kill(X,X).

:- peop(X), loc(X).
:- peop(X), org(X).
:- peop(X), other(X).
:- loc(X), org(X).
:- loc(X), other(X).
:- org(X), other(X).


% Triplets
orgbasedIn(O, P2) :- locatedIn(P1,P2), orgbasedIn(O,P1).
{locatedIn(P1, P2); locatedIn(P2, P1)} :- orgbasedIn(O,P1), orgbasedIn(O,P2).


locatedIn(P1, P3) :- locatedIn(P1,P2), locatedIn(P2,P3).


liveIn(X,P2) :- liveIn(X,P1), locatedIn(P1,P2).
{locatedIn(P1, P2); locatedIn(P2, P1)} :- liveIn(X,P1), liveIn(X,P2).


% Not absolutely correct, we assume no one works remotely
liveIn(X,P) :- workFor(X,O), orgbasedIn(O,P).
orgbasedIn(O,P) :- liveIn(X,P), workFor(X,O).

% Facts
