#script (python) 

import datetime
import string
import sys 
import clingo 
import os

global curr_as 
global maxLength 
global listAll 
global debug 

def process_model(m):
    global curr_as
    global listAll 

    curr_as = m.symbols(atoms=True)  
    
    if (debug): print ("List of all elements: {}".format(curr_as))        


def main(prg):

    global listAll   
    global debug 
    
    debug = False 
    listAll = []
    counter = 0 
    maxLength = 0 

    # compile the program 
        
    prg.ground([("base",[])]) 

    # get first answer set, process  

    while (prg.solve(None, on_model=process_model).satisfiable):  
   
        print("Answer set # ", counter)  
    
        keep_atoms = [x for x in curr_as if x.match("ok", 1)] 
    
        if (debug) :  print ("List of all OK elements *********** : {}".format(keep_atoms))  

        # can check whether the current answer set (keep_atoms) is a subset of some other answer set 
        # in listAll then discarded it
        # for now, simplifying by just see if the size of it is smaller than the max then will not add it 
        # listAll is a list of lists - so checking containment is not too difficult 
    
        if len(keep_atoms) >= maxLength : 
                 maxLength =  len(keep_atoms)
                 listAll.append(keep_atoms) 
    
        if (debug): print ("Current list of all answer sets", len(listAll))  
        
        constraint = ":-" + ''.join([str(x)+"," for x in keep_atoms]) + " 1==1."
    
        if (debug):  print("Constraint ... \n ",  constraint) 

        prg.add("constraints", [],  constraint)
    
        prg.ground([("constraints",[])])

        # ret = prg.solve(None, on_model=process_model)
     
        counter = counter + 1 
     
    #print("\n\n All optimal answer sets:",  listAll)
    newListAll = []
    for answerset in listAll:
        _list = []
        for atom in answerset:
            atom = str(atom)
            _list.append(atom)
        newListAll.append(_list)
    print(newListAll)

     
#end.
 

