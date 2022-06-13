% Program …
% tcson62@gmail.com

people(X) :- ok(liveIn(X, _)). %%%, liveIn(X, _).

:- liveIn(X, Y), people(X), people(Y).



Set of facts — ML algorithm : people(peter), b, c, ….

NS: atom(people(peter)), ….

{ok(X)} 1 :- atom(X).

nOfOKAtoms(N) :- N = #count{1,X : ok(X)}.

#maximize {N : nOfOKAtoms(N)}.