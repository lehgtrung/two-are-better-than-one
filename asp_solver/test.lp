
% There are two ways to model the problem
I. Model the problem as a table (like in sudoku)
% 0. Initialize the table (with model's prediction)

% 1. Define the grid


% 2. Cell type constraints
% 2.1 Type constraints on the diagonal line

% 2.2 Type constraints on the table's cells

% 3. Relation type constraints
    % If People(i,j) and Loc(n,m) then the grid [i-j,n-m] must have relation type of Live-In or No-Relation

% 4. Completion constraints
    % If a relation is symmetrical then if there is a forward
    % relationship at (i,j) then there has to be a backward relationship at (j,i).

    % If head entity (i-j) and tail entity (n-m) has a relationship at any cell in the grid ranging from [i-j,n-m]
    % then any other cell in that grid has to have the same relationship.

% 5. Transitional constraints
    % Live_In(i,k) <- Live_In(i,j) & Located_In(j,k) #Triplet rule
    % Live_In(t,y) <- Live_In(x,y) & Live_In(x,z) & Live_In(t,z) #Quadruplet rule

% 6. Grammatical constraints
    % TBD

II. Model the problem as tokens
% We generate all tokens upto a pre-defined length
% 'For singer and guitarist Bob Boyd , 57 , the Ranch has been home since he left Chattanooga , Tenn. , 35 years ago .'
% -> token('For'), token('singer'), token('For singer and guitarist'),..., token('For singer ... 35 years ago')
% Facts: Peop('Bob Boyd'), Loc('Tenn'),...
% Rules the same as above



We used 30% labeled data
Example
Sentence = "But Sen. Strom Thurmond , R-S.C. , sponsor of a proposed constitutional amendment , said ,
` ` An overwhelming majority of Americans are looking to the Congress to ...
restore the proper civil respect for our American flag. ' '"
Ground truth:
"Strom Thurmond": People
"Americans": Other
"Congress": Organization
***
"Strom Thurmond" WorkFor "Congress"
------------------------------------
Model prediction
"Strom Thurmond": People
"R-S.C": Location
"Congress": Organization
***
"Strom Thurmond" LiveIn "R-S.C"
"Congress" OrgBasedIn "R-S.C"
------------------------------------


Sentence = 'KAL attorney George Tompkins countered that the main evidence in the case
is a line drawn by Soviet officials on a map showing an off-course Flight 007 flying for several
hours over Russian airspace hundreds of miles to the west of its planned route.'
Ground truth:
"KAL": Org
"George Tompkins": People
"Soviet": Location
"Flight 007": Other
***
"George Tompkins" WorkFor "KAL"
---------------------------------------
Model prediction
"KAL": Org
"George Tompkins": People
"Soviet": Location
***
"KAL" LocatedIn "Soviet"
"George Tompkins" WorkFor "KAL"
"George Tompkins" LiveIn "Soviet"

"KAL" LocatedIn "Soviet" &  "George Tompkins" WorkFor "KAL" & "George Tompkins" LiveIn "Soviet" => test against KB
"KAL" LocatedIn "Soviet" &  "George Tompkins" WorkFor "KAL"
"KAL" LocatedIn "Soviet" & "George Tompkins" LiveIn "Soviet"

KB: False <- LocatedIn(x,y), Peop(x)


Sentence = 'In 1882 , the notorious outlaw Jesse James was shot to death in St. Joseph , Mo. ,
by Robert Ford , a member of his own gang .'
Ground truth:
"Jesse James": People
"St. Joseph , Mo.": Location
"Robert Ford": People
***
"Robert Ford" Kill "Jesse James"
------------------------------------------
Model prediction
"Jesse James": People
"St. Joseph , Mo.": Location
"Robert Ford": People
***
"Jesse James" LiveIn "St. Joseph , Mo."
"Robert Ford" LiveIn "St. Joseph , Mo."
