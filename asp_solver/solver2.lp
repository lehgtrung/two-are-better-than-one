
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

%peop("Prince_Ranariddh").
%loc("Phnom_Penh").
%loc("Jakarta").
%loc("Cambodia").
%locatedIn("Prince_Ranariddh","Jakarta").
%liveIn("Prince_Ranariddh","Phnom_Penh").


%loc("Benton").
%loc("Tenn.").
%org("Tennessee_Tech").
%loc("Cookville").
%loc("Virginia_Tech").
%locatedIn("Benton","Tenn.").
%locatedIn("Tennessee_Tech","Cookville").
%locatedIn("Cookville","Tenn.").


%peop("Woodrow_Wilson").
%loc("Virginia").
%loc("New_Jersey").
%other("Southerner").
%liveIn("Woodrow_Wilson","Virginia").
%liveIn("Woodrow_Wilson","New_Jersey").

%peop("Harrington").
%org("Socialist_International").
%loc("Stockholm").
%loc("Sweden.").
%workFor("Harrington","Socialist_International").
%orgbasedIn("Socialist_International","Stockholm").
%orgbasedIn("Socialist_International","Sweden.").
%locatedIn("Stockholm","Sweden.").

loc("U.S.").
loc("Havana").
org("Radio_Reloj_Network").
other("2100_GMT").
other("13_Feb_94").
orgbasedIn("Radio_Reloj_Network","Havana").
orgbasedIn("Radio_Reloj_Network","U.S.").