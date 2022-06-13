
% Type requirements
% peop(X), loc(P) :- liveIn(X,P).
% loc(P), loc(Q) :- locatedIn(P,Q).
% org(O), loc(P) :- orgbasedIn(O,P).
% peop(X), org(O) :- workFor(X,O).
% peop(X), peop(Y):- kill(X,Y).

ok(peop(X)) :- ok(liveIn(X,P)).
ok(loc(P)) :- ok(locatedIn(P,Q)).
ok(org(O)) :- ok(orgbasedIn(O,P)).
ok(peop(X)) :- ok(workFor(X,O)).
ok(peop(X)) :- ok(kill(X,Y)).

ok(loc(P)) :- ok(liveIn(X,P)).
ok(loc(Q)) :- ok(locatedIn(P,Q)).
ok(loc(P)) :- ok(orgbasedIn(O,P)).
ok(org(O)) :- ok(workFor(X,O)).
ok(peop(Y)):- ok(kill(X,Y)).

% Relation symmetricity
% {liveIn(X,P); liveIn(P,X)}.
% {locatedIn(P,Q); locatedIn(Q,P)}.
% {orgbasedIn(O,P); orgbasedIn(P,O)}.
% {workFor(X,O); workFor(O,X)}.
% {kill(X,Y); kill(Y,X)}.


% Other requirements
:- ok(liveIn(X,X)).
:- ok(locatedIn(X,X)).
:- ok(orgbasedIn(X,X)).
:- ok(workFor(X,X)).
:- ok(kill(X,X)).

:- ok(peop(X)), ok(loc(X)).
:- ok(peop(X)), ok(org(X)).
:- ok(peop(X)), ok(other(X)).
:- ok(loc(X)), ok(org(X)).
:- ok(loc(X)), ok(other(X)).
:- ok(org(X)), ok(other(X)).

:- atom(org(X)), not ok(org(X)), ok(locatedIn(X, Y)). 
:- atom(org(X)), ok(org(X)), #count{1,Y : ok(locatedIn(X, Y))}==0.

% Triplets
ok(orgbasedIn(O, P2)) :- ok(locatedIn(P1,P2)), ok(orgbasedIn(O,P1)).
{ok(locatedIn(P1, P2)); ok(locatedIn(P2, P1))} :- ok(orgbasedIn(O,P1)), ok(orgbasedIn(O,P2)).


ok(locatedIn(P1, P3)) :- ok(locatedIn(P1,P2)), ok(locatedIn(P2,P3)).


ok(liveIn(X,P2)) :- ok(liveIn(X,P1)), ok(locatedIn(P1,P2)).
{ok(locatedIn(P1, P2)); ok(locatedIn(P2, P1))} :- ok(liveIn(X,P1)), ok(liveIn(X,P2)).


% Not absolutely correct, we assume no one works remotely
ok(liveIn(X,P)) :- ok(workFor(X,O)), ok(orgbasedIn(O,P)).
ok(orgbasedIn(O,P)) :- ok(liveIn(X,P)), ok(workFor(X,O)).

% Facts

atom(org("CDT_Tuesday")).
atom(loc("Port_Arther")).
atom(loc("Galveston")).
atom(loc("Texas.")).
atom(locatedIn("Galveston","Texas.")).
atom(locatedIn("CDT_Tuesday","Texas.")).
atom(locatedIn("Port_Arther","Texas.")).
atom(locatedIn("Port_Arther","Galveston")).

{ ok(X) } :- atom(X). 

nOfOKAtoms(N) :- N = #count{1,X : ok(X),atom(X)}. 

#maximize {N : nOfOKAtoms(N)}. 

%#show ok/1.
%#show nOfOKAtoms/1.
 