# Compute the honking protocol for the exam cheaters
import math
import random

def compute_and_send_code(exam):
    code = [0] * 10
    # Dont change anything above this line
    # ==========================

    # TODO Add your solution here.

    for x in range(0, len(exam)-5, 3):

        list = [exam[x], exam[x+1], exam[x+2]] 
           
        bit = max(list,key=list.count)
        if(bit == 0):
            code[round(x/3)]=0
        else:
            code[round(x/3)]=1
        bit = None
        list = None    
        

    for y in range(15,len(exam),1):
        if exam[y]==0:
            code[y-10]=0
        else:
            code[y-10]=1

    # ==========================
    # Dont change anything below this line
    return code


def enter_solution_based_on_code(code):
    answer = [0] * 20

    # Dont change anything above this line
    # ==========================

    # TODO Add your solution here.
    for x in range(0, len(code)-5, 1):
        if code[x]==0:
            for a in range (3):
                answer[(x*3) + a ] = 0
        
        elif code[x]==1:
            for a in range (3):
                answer[(x*3) + a ] = 1
    
    for y in range(5, len(code), 1):
        if code[y]==0:
            answer[y+10]=0
        else:
            answer[y+10]=1
    # ==========================
    # Dont change anything below this line
    return answer