
% Type requirements
peop(X), loc(P) :- liveIn(X,P).
loc(P), loc(Q) :- locatedIn(P,Q).
org(O), loc(P) :- orgBasedIn(O,P).
peop(X), org(O) :- workFor(X,O).
peop(X), peop(Y):- kill(X,Y).


% Relation symmetricity
% {liveIn(X,P); liveIn(P,X)}.
% {locatedIn(P,Q); locatedIn(Q,P)}.
% {orgBasedIn(O,P); orgBasedIn(P,O)}.
% {workFor(X,O); workFor(O,X)}.
% {kill(X,Y); kill(Y,X)}.


% Other requirements
:- liveIn(X,X).
:- locatedIn(X,X).
:- orgBasedIn(X,X).
:- workFor(X,X).
:- kill(X,X).

:- peop(X), loc(X).
:- peop(X), org(X).
:- peop(X), other(X).
:- loc(X), org(X).
:- loc(X), other(X).
:- org(X), other(X).


% Triplets
orgBasedIn(O, P2) :- locatedIn(P1,P2), orgBasedIn(O,P1).
{locatedIn(P1, P2); locatedIn(P2, P1)} :- orgBasedIn(O,P1), orgBasedIn(O,P2).


locatedIn(P1, P3) :- locatedIn(P1,P2), locatedIn(P2,P3).


liveIn(X,P2) :- liveIn(X,P1), locatedIn(P1,P2).
{locatedIn(P1, P2); locatedIn(P2, P1)} :- liveIn(X,P1), liveIn(X,P2).


liveIn(X,P) :- workFor(X,O), orgBasedIn(O,P). % Not absolutely correct, we assume no one works remotely
% workFor(X,O) :- liveIn(X,P), orgBasedIn(O,P). % not true
orgBasedIn(O,P) :- liveIn(X,P), workFor(X,O). % Not absolutely correct, we assume no one works remotely


% Quadruplets
% liveIn(Y,P2) :- liveIn(X,P1), liveIn(X,P2), liveIn(Y,P1), locatedIn(P1,P2).
% locatedIn(R,Q2) :- locatedIn(P,Q1), locatedIn(P,Q2), locatedIn(R,Q1), locatedIn(Q1,Q2).
% orgBasedIn(O2,P2) :- orgBasedIn(O1,P1), orgBasedIn(O1,P2), orgBasedIn(O2,P1), locatedIn(P1,P2).


% liveIn(X,P2) :- liveIn(X,P1), orgBasedIn(O,P1), orgBasedIn(O,P2), locatedIn(P1,P2).
% liveIn(Y,P1) :- workFor(X,O1), orgBasedIn(O1,P1), workFor(Y,O1).


% orgBasedIn(O,P2) :- liveIn(X,P1), liveIn(X,P2), orgBasedIn(O,P1), locatedIn(P1,P2). % This is the same as rule #1
% workFor(Y,O2) :- workFor(X,O1), workFor(X,O2), workFor(Y,O1).
% Alice works for department of States and John works for US government
% Bob works for department of defense -> Bob works for US government?
% We need relation between O1 and O2
% The same happens with relation Kill
% kill(Z,Y2) :- kill(X,Y1), kill(X,Y2), kill(Z,Y1).


% Facts

peop("George_Shibley").
peop("Robert_F._Kennedy").
peop("Sirhan_Sirhan").
% locatedIn("Sirhan_Sirhan","Robert_F._Kennedy").
