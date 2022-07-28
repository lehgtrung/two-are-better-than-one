% Type requirements 
:- ok(locatedIn(P,Q)), ok(org(P)).
:- ok(locatedIn(P,Q)), ok(peop(P)).
:- ok(locatedIn(P,Q)), ok(other(P)).
:- ok(locatedIn(P,Q)), ok(org(Q)).
:- ok(locatedIn(P,Q)), ok(peop(Q)).
:- ok(locatedIn(P,Q)), ok(other(Q)).

:- ok(orgbasedIn(P,Q)), ok(loc(P)).
:- ok(orgbasedIn(P,Q)), ok(peop(P)).
:- ok(orgbasedIn(P,Q)), ok(other(P)).
:- ok(orgbasedIn(P,Q)), ok(org(Q)).
:- ok(orgbasedIn(P,Q)), ok(peop(Q)).
:- ok(orgbasedIn(P,Q)), ok(other(Q)).

:- ok(liveIn(P,Q)), ok(loc(P)).
:- ok(liveIn(P,Q)), ok(org(P)).
:- ok(liveIn(P,Q)), ok(other(P)).
:- ok(liveIn(P,Q)), ok(peop(Q)).
:- ok(liveIn(P,Q)), ok(org(Q)).
:- ok(liveIn(P,Q)), ok(other(Q)).

:- ok(workFor(P,Q)), ok(loc(P)).
:- ok(workFor(P,Q)), ok(org(P)).
:- ok(workFor(P,Q)), ok(other(P)).
:- ok(workFor(P,Q)), ok(peop(Q)).
:- ok(workFor(P,Q)), ok(loc(Q)).
:- ok(workFor(P,Q)), ok(other(Q)).

:- ok(kill(P,Q)), ok(loc(P)).
:- ok(kill(P,Q)), ok(org(P)).
:- ok(kill(P,Q)), ok(other(P)).
:- ok(kill(P,Q)), ok(loc(Q)).
:- ok(kill(P,Q)), ok(org(Q)).
:- ok(kill(P,Q)), ok(other(Q)).


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


% Facts

%atom(org("CDT_Tuesday")).
%atom(loc("Port_Arther")).
%atom(loc("Galveston")).
%atom(loc("Texas.")).
%atom(locatedIn("Galveston","Texas.")).
%atom(locatedIn("CDT_Tuesday","Texas.")).
%atom(locatedIn("Port_Arther","Texas.")).
%atom(locatedIn("Port_Arther","Galveston")).


atom(org("Knight-Ridder_Inc.")).
atom(loc("Pasadena")).
atom(loc("Calif.")).
atom(org("Post-Tribune")).
atom(loc("Gary")).
atom(loc("Ind")).
atom(locatedIn("Knight-Ridder_Inc.","Pasadena")).
atom(locatedIn("Pasadena","Calif.")).
atom(orgbasedIn("Post-Tribune","Gary")).
atom(locatedIn("Post-Tribune","Calif.")).
atom(locatedIn("Knight-Ridder_Inc.","Ind")).
atom(locatedIn("Knight-Ridder_Inc.","Gary")).
atom(orgbasedIn("Knight-Ridder_Inc.","Calif.")).
atom(locatedIn("Gary","Ind")).
atom(orgbasedIn("Post-Tribune","Ind")).


{ ok(X) } :- atom(X).

nok(X) :- atom(X), not ok(X). 

nOfOKAtoms(N) :- N = #count{1,X : ok(X),atom(X)}.

#maximize {N : nOfOKAtoms(N)}.

#show ok/1.
#show nOfOKAtoms/1.


