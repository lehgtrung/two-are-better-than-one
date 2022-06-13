
% Type requirements

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

% :- ok(locatedIn(P,Q)), not ok(loc(P))


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

% LocatedIn
%:- atom(org(X)), not ok(org(X)), ok(locatedIn(X, Y)).
%:- atom(org(X)), ok(org(X)), #count{1,Y : ok(orgbasedIn(X, Y))}==0.


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

#show ok/1.
#show nOfOKAtoms/1.
