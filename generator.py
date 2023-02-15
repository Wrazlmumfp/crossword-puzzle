#!/usr/bin/env python3

import numpy as np
import random
import time
import string
import copy


class crossword:
    def __init__(self):
        # letters[(1,5)] == "R"
        self.letters = dict()
        # wordsHor[(6,4)] == "WORD"
        self.wordsHor = dict()
        # wordsVer[(7,1)] == "WORD"
        self.wordsVer = dict()
        # clues["WORD"] == "Some word."
        self.clues = dict()
        # set of all coordinates where two words intersect
        self.intersections = set()
        # solutionDict[(1,5)] == "B"
        # where "B" is the index of the letter at (1,5) in the solution, NOT the letter at position (1,5)
        self.solutionDict = dict()
        self.solution = None
        self.xMin = 0
        self.xMax = 0
        self.yMin = 0
        self.yMax = 0
        # undirected graph with letter coordinates as nodes and neighbouring letters as edges
        self.graph = graph()
    def __getitem__(self,i):
        return self.get(i)
    def get(self,x,y=None):
        if y == None: # i is a tuple
            x,y = x
        try:
            return self.letters[x,y]
        except KeyError:
            return " "
    def getSolution(self):
        return self.solution
    def set(self,x,y,l):
        self.letters[x,y] = l
    def numRows(self):
        return self.yMax-self.yMin+1
    def numCols(self):
        return self.xMax-self.xMin+1
    def numLetters(self):
        return len(self.letters)
    def numWords(self):
        return len(self.wordsHor)+len(self.wordsVer)
    def numIntersections(self):
        return len(self.intersections)
    def numCycles(self):
        global args
        if not(args.cycles):
            return 0
        else:
            return self.graph.numCycles()
    def __str__(self):
        n = self.numRows()
        m = self.numCols()
        L = [ [" " for i in range(m)] for i in range(n) ]
        for d in self.letters.keys():
            # bold if solution letter
            L[n-(d[1]-self.yMin+1)][d[0]-self.xMin] = self.get(d) if not(d) in self.solutionDict.keys() else "\033[1m"+self.get(d)+"\033[0m"
        outputLines = ["".join([str(i) for i in range(max(self.xMin,-9),0)])+" "+" ".join([str(i) for i in range(0,min(self.xMax+1,10))])] \
                      + [chr(9484)+chr(9472)*(2*m-1) + chr(9488)] \
                      + [chr(9474)+" ".join(row)+chr(9474) for row in L] \
                      + [chr(9492)+chr(9472)*(2*m-1)+chr(9496)]
        for j in range(2,n+2):
            outputLines[j] = outputLines[j] + " " + str(self.yMax-j+2)
        return "\n".join(outputLines)
    def __repr__(self):
        return str(self)
    def latexPuzzle(self):
        n = self.numRows()
        m = self.numCols()
        L = [ ["{}" for i in range(m)] for i in range(n) ]
        # insert letters
        for d in self.letters.keys():
            if d in self.solutionDict.keys():
                L[n-(d[1]-self.yMin+1)][d[0]-self.xMin] = r"[\cwText{"+self.solutionDict[d]+"}][gfo]{"+self.get(d)+"}"
            else:
                L[n-(d[1]-self.yMin+1)][d[0]-self.xMin] = "[][gf]{"+self.get(d)+"}"
        wordsHor_sorted = list(self.wordsHor.keys())
        wordsHor_sorted.sort(key=lambda x: -x[1])
        wordsVer_sorted = list(self.wordsVer.keys())
        wordsVer_sorted.sort(key=lambda x: x[0])
        words_sorted = wordsHor_sorted + wordsVer_sorted # note: not words, but coordinates to words!
        words_bothHorVer = [c for c in wordsHor_sorted if c in wordsVer_sorted]
        # insert numbers for boxes with both horizontal and vertical word starts
        for d in words_bothHorVer:
            indices = [i for i,x in enumerate(words_sorted) if x == d] # should be two
            if d in self.solutionDict.keys():
                # overwrites previously written letter
                # e.g. [\cwNumText{21/42}{B}][o]{N}
                L[n-(d[1]-self.yMin+1)][d[0]-self.xMin] = r"[\cwNumText{"+str(indices[0]+1)+"/"+str(indices[1]+1)+"}{"+str(self.solutionDict[d])+"}][ogf]"+"{"+self.get(d)+"}"
            else:
                # e.g. [\cwNum{42}][B]
                L[n-(d[1]-self.yMin+1)][d[0]-self.xMin] = "["+str(indices[0]+1)+"/"+str(indices[1]+1)+"][gf]{"+self.get(d)+"}"
        # insert all other word numbers
        for i,d in enumerate(words_sorted):
            if d in words_bothHorVer:
                continue
            if d in self.solutionDict.keys():
                L[n-(d[1]-self.yMin+1)][d[0]-self.xMin] = r"[\cwShortNumText{"+str(i+1)+"}{"+str(self.solutionDict[d])+"}][gfo]{"+self.get(d)+"}"
            else:
                L[n-(d[1]-self.yMin+1)][d[0]-self.xMin] = "["+str(i+1)+"][gf]{"+self.get(d)+"}"
        return r"\begin{Puzzle}{"+str(self.numCols())+"}{"+str(self.numRows())+"}%\n  |" \
            +"|.\n  |".join([" |".join(l) for l in L]) + "\n\\end{Puzzle}"
    def latexClues(self):
        wordsHor_sorted = list(self.wordsHor.keys())
        wordsHor_sorted.sort(key=lambda x: -x[1])
        wordsVer_sorted = list(self.wordsVer.keys())
        wordsVer_sorted.sort(key=lambda x: x[0])
        hor = "{\\Large \\textbf{Horizontal}}\n" \
            +"\\begin{multicols}{2}\n" \
            +"  \\begin{enumerate}\n" \
            +"\n".join(["  \\item[("+str(i+1)+")] "+self.clues[self.wordsHor[d]] for i,d in enumerate(wordsHor_sorted)]) \
            +"\n  \\end{enumerate}" \
            +"\n\\end{multicols}"
        ver ="\n{\\Large \\textbf{Vertikal}}\n" \
            +"\\begin{multicols}{2}\n" \
            +"  \\begin{enumerate}\n" \
            +"\n".join(["  \\item[("+str(len(self.wordsHor)+i+1)+")] "+self.clues[self.wordsVer[d]] for i,d in enumerate(wordsVer_sorted)]) \
            +"\n  \\end{enumerate}" \
            +"\n\\end{multicols}"
        return  "\\begin{multicols}{2}\n" + hor + "\n\n\\columnbreak\n\n" + ver + "\n\\end{multicols}\n"
    def copy(self):
        """Returns a copy of self."""
        global args
        other = crossword()
        other.letters = self.letters.copy()
        other.wordsHor = self.wordsHor.copy()
        other.wordsVer = self.wordsVer.copy()
        other.clues = self.clues.copy()
        other.intersections = self.intersections.copy()
        other.solutionDict = self.solutionDict.copy()
        other.solution = self.getSolution()
        other.xMin = self.xMin
        other.xMax = self.xMax
        other.yMin = self.yMin
        other.yMax = self.yMax
        if args.cycles:
            other.graph = self.graph.copy()
        return other
    def add(self,word,x,y,hor=True,clue=None):
        """Adds word to self, first letter at position (x=col,y=row); hor is a bool whether word is inserted horicontal or vertical"""
        """Does not check whether word is addable (use self.isAddable before call if necessary)"""
        global args
        word = word.upper()
        self.clues[word] = clue
        if hor:
            self.wordsHor[x,y]=word
        else:
            self.wordsVer[x,y]=word
        self.xMin = min(self.xMin,x)
        self.yMin = min(self.yMin,y) if hor else min(self.yMin,y-len(word)+1)
        self.xMax = max(self.xMax,x+len(word)-1) if hor else max(self.xMax,x)
        self.yMax = max(self.yMax,y)
        for k,l in enumerate(word):
            coord = (x+k,y) if hor else (x,y-k)
            prev = (x+k-1,y) if hor else (x,y-k+1) # coordinate before coord
            if coord in self.letters.keys():
                self.intersections.add(coord)
            self.set(coord[0],coord[1],l)
            if not(args.cycles):
                continue
            self.graph.add_node(coord)
            if k > 0:
                self.graph.add_edge(prev,coord)
        return
    def isAddable(self,word,x,y,hor=True):
        """Throws an error if word can not be added"""
        word = word.upper()
        relevantKeys = self.letters.keys()
        for k in range(len(word)):
            coords = (x+k,y) if hor else (x,y-k)
            left = (x+k,y+1) if hor else (x+1,y-k)
            right = (x+k,y-1) if hor else (x-1,y-k)
            # checks whether letter would overwrite other letter
            if not(self.get(coords) in {word[k]," "}):
                raise Exception("Cannot add word "+word+" at position "+str((x,y))+" because of letter "+self.get(coords)+" instead of "+word[k]+" at position "+str(coords)+".")
            # checks whether neighbouring positions are only letters of intersections words
            if not(self.get(coords) == word[k]) and {self.get(left),self.get(right)} != {" "}:
                raise Exception("Cannot add word "+word+" at position "+str((x,y))+" because it would neighbour other word.")
            # checks whether position is already taken by two words
            if coords in self.intersections:
                raise Exception("Cannot add word "+word+" at position "+str((x,y))+" because already two words intersect in letter "+str(word[k])+" at position "+str((x+k,y)))
            # checks whether letter is not beginning of other word
            wordDict = self.wordsHor if hor else self.wordsVer
            if coords in wordDict.keys():
                raise Exception("Cannot add word "+word+" at position "+str((x,y))+" as it would intersect with word "+wordDict[coords]+" at position "+str(coords)+".")
            # checks whether word runs parallel to other word
            if hor and k < len(word):
                if ((x+k,y+1) in relevantKeys and (x+k+1,y+1) in relevantKeys)\
                   or ((x+k,y-1) in relevantKeys and (x+k+1,y-1) in relevantKeys):
                    raise Exception("Cannot add word "+word+" at position "+str((x,y))+" as it would run parallel to another word.")
            elif not(hor) and k < len(word):
                if ((x+1,y-k) in relevantKeys and (x+1,y-k-1) in relevantKeys)\
                   or ((x-1,y-k) in relevantKeys and (x-1,y-k-1) in relevantKeys):
                    raise Exception("Cannot add word "+word+" at position "+str((x,y))+" as it would run parallel to another word.")
        nextCoord = (x+len(word),y) if hor else (x,y-len(word))
        prevCoord = (x-1,y) if hor else (x,y+1)
        # checks whether there is a letter before or after the word
        if nextCoord in relevantKeys or prevCoord in relevantKeys:
            raise Exception("Cannot add word "+word+" at position "+str((x,y))+" as there is a new letter after its end.")
        return
    def isAddableBool(self,word,x,y,hor=True):
        """Same as isAddable, but returns Boolean whether word can be added."""
        try:
            self.isAddable(word,x,y,hor)
            return True
        except:
            return False
    def whereIsAddable(self,word):
        word=word.upper()
        possibleCoordinates = set()
        for k in range(len(word)):
            l = word[k]
            coordinates = {j for j in self.letters.keys() if self.get(j) == l}
            for x,y in coordinates:
                if self.isAddableBool(word,x-k,y,True):
                    possibleCoordinates.add( (x-k,y,True) )
                elif self.isAddableBool(word,x,y+k,False):
                    possibleCoordinates.add( (x,y+k,False) )
        return possibleCoordinates
    def score(self,maxSizeX=None,maxSizeY=None):
        # euclidean norm
        return np.sqrt(sum([s**2 for s in self.scores(maxSizeX,maxSizeY)]))
    def scores(self,maxSizeX=None,maxSizeY=None):
        if self.numWords() == 0:
            return (0,0)
        if maxSizeX != None and self.numCols() > maxSizeX:
            return (0,0)
        if maxSizeY != None and self.numRows() > maxSizeY:
            return (0,0)
        # (many cycles, many intersections, few columns, few rows, wordsHor/wordsVer balanced)
        return (
            1/4*self.numWords()*self.numCycles(),
            3*self.numIntersections(),
            self.numWords()**2*1/self.numCols(),
            self.numWords()**2*1/self.numRows(),
            1/8*len(self.wordsHor)*len(self.wordsVer)
        )
    def setSolution(self,solution):
        global alphabet
        global args
        if self.solution != None:
            raise Exception("Crossword already has a solution.")
        alreadyTaken = set()
        solutionDict = dict() # self.solutionDict is not touched until it is clear that the solution is possible
        for i,l in enumerate(solution.upper().replace(" ","")):
            possibleCoords = [coord for coord in self.letters.keys() if self.letters[coord] == l and not(coord in alreadyTaken)]
            if possibleCoords == []:
                return False
            position = random.sample(possibleCoords,1)[0]
            solutionDict[position] = alphabet[i]
            alreadyTaken.add(position)
        self.solutionDict = solutionDict
        self.solution = solution
        return True
    def generate(wordDict,sentenceDict,maxSizeX=None,maxSizeY=None):
        """Generates an n x m crossword puzzle from list words if possible"""
        words = list(wordDict.keys())
        words.sort(key=len)
        # place longest word in the middle of the crossword
        if not(args.quiet):
            print("Doing iteration Nr. " + str(i),end="\r")
        c = crossword()
        w = random.choices(words,list(range(1,len(words)+1)))[0]
        if w in sentenceDict.keys():
            words = [word for word in words if not(word in sentenceDict[w])]
        else:
            words.remove(w)
        hor = bool(random.randint(0,1))
        c.add(w,0,0,hor,wordDict[w])
        added =[(w,(0,0,hor))]
        sthChanged = True
        laidAside = []
        while sthChanged:
            sthChanged = False
            while len(words) > 0:
                # choose random word from list
                # longer words are chosen with higher probability
                #w = random.choices(words,list(range(1,len(words)+1)))[0]
                #w = random.choices(words)[0]
                w = random.choices(words,[np.log(len(wort)+1) for wort in words])[0]
                words.remove(w)
                coords = c.whereIsAddable(w)
                # newCWs is list of tuples (new crossword, word added to this new crossword)
                newCWs = list()
                for coord in coords:
                    cNew = c.copy()
                    cNew.add(w,coord[0],coord[1],coord[2],wordDict[w])
                    if cNew.score(maxSizeX,maxSizeY) > 0:
                        newCWs.append((cNew,w,coord))
                if len(newCWs) == 0:
                    laidAside.append(w)
                    continue
                sthChanged = True
                if w in sentenceDict.keys():
                    laidAside = [word for word in laidAside if not(word in sentenceDict[w])]
                    words = [word for word in words if not(word in sentenceDict[w])]
                newCWs.sort(key=lambda x: x[0].score(maxSizeX,maxSizeY))
                newCW = random.choices(newCWs,range(1,len(newCWs)+1))[0]
                c = newCW[0]
                added.append(newCW[1:])
            words = laidAside
            laidAside = []
        #print("added in order: " + str(added))
        return c
    

    
class graph:
    # undirected graph
    def __init__(self):
        # nodes should be set of tuples (coordinates), edges set of frozensets of tuples
        self.nodes = set()
        self.edges = set()
        # important for graph.numCycles()
        self.parent = dict()
        self.color = dict()
        self.cycleNumber = None
        # dictionary storing all neighbours of nodes
        self.neighbours = dict()
    def __str__(self):
        return "graph("+str(self.nodes)+", {"+", ".join([str(tuple(e)) for e in self.edges])+"})"
    def __repr__(self):
        return str(self)
    def add_node(self,node):
        if not(node in self.nodes):
            self.nodes.add(node)
            self.neighbours[node] = set()
    def extend_nodes(self,L):
        for l in L:
            self.add_node(l)
    def remove_node(self,node):
        self.nodes.discard(node)
        del self.neighbours[node]
    def remove_nodes(self,L):
        for node in L:
            self.remove_node(node)
    def add_edge(self,node1,node2):
        self.edges.add( frozenset([node1, node2]) )
        self.neighbours[node1].add(node2)
        self.neighbours[node2].add(node1)
    def extend_edges(self,L):
        for node1,node2 in L:
            self.add_edge(node1,node2)
    def remove_edge(self,node1,node2):
        self.edges.discard( frozenset([node1, node2]) )
        self.neighbours[node1].discard(node2)
        self.neighbours[node2].discard(node1)
    def remove_edges(self,L):
        for node1,node2 in L:
            self.remove_edge(node1,node2)
    def copy(self):
        other = graph()
        other.nodes = self.nodes.copy()
        other.edges = self.edges.copy()
        other.neighbours = {node: self.neighbours[node].copy() for node in self.nodes}
        other.parent = dict()
        other.color = dict()
        other.cycleNumber = None
        return other
    def getNeighbours(self,u):
        return {v for v in self.nodes if frozenset([u,v]) in self.edges}
    def numCycles(self):
        # for details about the algorithms, see https://www.codingninjas.com/codestudio/library/count-of-simple-cycles-in-a-connected-undirected-graph-having-n-vertices
        self.cycleNumber = 0
        for v in self.nodes:
            self.color[v] = None
            self.parent[v] = None
        # (0,0) is the start node; should be in any graph handled here
        self.DFSCycle((0,0),0)
        return self.cycleNumber
    def DFSCycle(self,u,p):
        # u is the currently visited node, p its parent we are coming from
        # the node is already considered
        if self.color[u] == 2:
            return
        # partially visited node found i.e new cycle found
        if self.color[u] == 1:
            self.cycleNumber += 1
            return
        # storing parent of u
        self.parent[u] = p
        # marking as partially visited
        self.color[u] = 1
        for v in self.neighbours[u]:
            if v == self.parent[u]:
                continue
            self.DFSCycle(v, u)
        # marking as fully visited
        self.color[u] = 2
        return



alphabet = list(string.ascii_uppercase)+list(string.ascii_lowercase)+list("0123456789")

# replaces ä by ae and so on
def convertWord(word):
    return word.upper().replace("Ä","AE").replace("Ö","OE").replace("Ü","UE").replace("ß","SS")


def latex(c,title,subtitle,info):
    return printPreamble()+"\n" \
        + "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n" \
        + "%%%%%%%%%%%%%%%%%%%   Document   %%%%%%%%%%%%%%%%%%%\n" \
        + "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n" \
        + "\\begin{document}"+printHeader(title,subtitle)+"\n\n\n"+printPuzzleDefs()+"\n\n" \
        + c.latexPuzzle() \
        + "\n\n\\vspace{0.5cm}\n\n" \
        + printSolution(c.solution) \
        + "\n\n\\newpage\n" \
        + "\\begin{landscape}\n\\scriptsize\n" \
        + "\n".join(info)+"\\par\\bigskip\n\n" \
        + c.latexClues() \
        + "\n\\end{landscape}" \
        + "\n\n\end{document}"
    


def printPuzzleDefs():
    return r"""
\PuzzleDefineColorCell{g}{cellcolor}            % g option for gray cells
\def\PuzzleNumberFont{\rmfamily\size{6}}        % size of small numbers and letters in the corners
\def\PuzzleSolutionContent#1{\makebox(1,1){#1}} % no automatic uppercase in solution
%\PuzzleSolution                                 % show solution"""

def printHeader(title,subtitle):
    return r"""
\begin{center}
  {\LARGE\textbf{"""+title+r"""}} \\
  {\textbf{"""+subtitle+r"""}} \\
\end{center}
\pagenumbering{gobble}
\RaggedRight
\par\bigskip
"""

def printSolution(solution):
    global alphabet
    global args
    if solution == None:
        return ""
    # split solution at spaces in parts of length <= 23
    solutionList = solution.split(" ")
    splittedSolution = []
    s = []
    for l in solutionList:
        if len("".join(s))+len(s)+len(l) > 23:
            splittedSolution.append(s)
            s = [l]
        else:
            s.append(l)
    splittedSolution.append(s)
    # splittedSolution is of the form ["HELLO I AM A", "SPLITTED SOLUTION"]
    splittedSolution = [" ".join(l) for l in splittedSolution]
    # to jump over spaces in the solution
    solution_alphabet = []
    i = 0
    for l in solution:
        if l == " ":
            solution_alphabet.append("")
        else:
            solution_alphabet.append(alphabet[i])
            i += 1
    sol="Solution" if args.english else "Lösung"
    return sol+""": \\par \\vspace{0.5cm}
\\begin{Puzzle}{23}{"""+str(2*len(splittedSolution)-1)+"}\n" \
    + "|.\n|.\n".join([
        " ".join([
            ("|[\\cwText{"
             + solution_alphabet[i+sum([len(sol_part)+1
                                       for sol_part in splittedSolution[:k]])]
             + "}][gfo]" + s)
            if s!=" "
            else "|{}"
            for i,s in enumerate(solution_part)
        ]) for k,solution_part in enumerate(splittedSolution)
    ]) \
            + """|.
\\end{Puzzle}
"""



def printPreamble():
    global args
    if args.nogray:
        cell_color = "1"
    else:
        cell_color = ".95"
    return r"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%   Preamble   %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\documentclass[a4paper,twoside,11pt]{article}

\usepackage{babel}
\usepackage[T1]{fontenc}                % Better wort separation
\usepackage[utf8]{inputenc}             % For äöüß etc.
\usepackage[margin=2.5cm]{geometry}     % Smaller margins
\usepackage{lmodern}                    % higher text resolution
\usepackage{enumerate}                  % Better enumerate definition
\usepackage{amsmath}                    % Makes everything better
\usepackage{amssymb}                    % for \mathbb, \mathfrak etc.
\usepackage{mathtools}                  % for \coloneqq etc.
\usepackage{cwpuzzle}                   % For crossword puzzles
\usepackage{ulem}                       % For \sout
\usepackage{xcolor}                     % For gray cells
\usepackage{pdflscape}                  % landscape
\usepackage{ragged2e}                   % left align of text
\usepackage{fancyhdr}                   % header
\usepackage{multicol}
\usepackage{tikz}

% small 3-digit hex number at top left to match puzzle with clues
\pagestyle{fancy}
\renewcommand{\headrulewidth}{0pt} % no ruler
\fancyhead[L]{\scriptsize\color{darkgray} """+hex(args.seed)[2:].upper()+r"""}

% shortcuts for crossword puzzle
\definecolor{cellcolor}{gray}{"""+cell_color+r"""} % background color of cells
\newcommand{\cwNumText}[2]{\tikz[overlay]{\filldraw[cellcolor] (0,0) rectangle (0.62,-0.2); \node[right] at (-0.1,-0.1) {#1}; \node[circle,fill=cellcolor] at (0.53,-0.53) {}; \node at (0.53,-0.53) {#2};}}
\newcommand{\cwShortNumText}[2]{\tikz[overlay]{\filldraw[cellcolor] (0,0) rectangle (0.31,-0.2); \node[right] at (-0.1,-0.1) {#1}; \node[circle,fill=cellcolor] at (0.53,-0.53) {}; \node at (0.53,-0.53) {#2};}}
\newcommand{\cwText}[1]{\tikz[overlay]{\node[circle,fill=cellcolor] at (0.53,-0.53) {}; \node at (0.53,-0.53) {#1};}}

% some other shortcuts
\newcommand{\Nat}{\mathbb{N}}
\newcommand{\Integ}{\mathbb{Z}}
\newcommand{\Real}{\mathbb{R}}
\newcommand{\Rat}{\mathbb{Q}}
\newcommand{\Comp}{\mathbb{C}}
\newcommand{\terms}{\mathbb{T}}
\newcommand{\field}{\mathbb{F}}
\newcommand{\im}{\operatorname{Im}}     	
\newcommand{\Ker}{\operatorname{Ker}}
\newcommand{\Hom}{\operatorname{Hom}}
\newcommand{\Supp}{\operatorname{Supp}}
\newcommand{\Ann}{\operatorname{Ann}}
\newcommand{\HF}{\operatorname{HF}}
\newcommand{\HP}{\operatorname{HP}}
\newcommand{\HS}{\operatorname{HS}}
\newcommand{\HN}{\operatorname{HN}}
\newcommand{\hn}{\operatorname{hn}}
\newcommand{\mult}{\operatorname{mult}}
\newcommand{\LT}{\operatorname{LT}}
\newcommand{\LM}{\operatorname{LM}}
\newcommand{\IG}{\mathcal{I}_G}
\newcommand{\MS}{\operatorname{MS}}
\newcommand{\LC}{\operatorname{LC}}
\newcommand{\GL}{\operatorname{GL}}
\newcommand{\NR}{\operatorname{NR}}
\newcommand{\id}{\operatorname{id}}
\newcommand{\End}{\operatorname{End}}
\newcommand{\ord}{\operatorname{ord}}
\newcommand{\rey}{\varrho}
\newcommand{\calA}{{\mathcal{A}}}
\newcommand{\isom}{\cong}
\newcommand{\calB}{{\mathcal{B}}}
\newcommand{\calC}{{\mathcal{C}}}
\newcommand{\calI}{{\mathcal{I}}}
\newcommand{\Rel}{{\operatorname{Rel}}}
\newcommand{\Mat}{{\operatorname{Mat}}}
\newcommand{\tr}{^{\rm tr}}
\newcommand{\trace}{{\rm trace}}
\renewcommand{\det}{{\rm det}}
\newcommand{\characteristic}{{\rm char}}
\newcommand{\rewrite}[1]{{\stackrel{#1}{\longrightarrow}_s}}
\newcommand{\rewriteequiv}[1]{{\stackrel{#1}{\longleftrightarrow}_s}}
\newcommand{\reductionstep}[1]{{\stackrel{#1}{\longrightarrow}_{ss}}}
\newcommand{\colour}[1]{\color{#1}}
% small matrices
\newcommand{\mat}[1]{\begin{psmallmatrix}#1\end{psmallmatrix}}

% font size
\newcommand{\size}[1]{\fontsize{#1}{#1}\selectfont{}}

%%%%%%%%%%%%%        No Indents      %%%%%%%%%%%%%%
\setlength{\parindent}{0mm}
"""



import argparse
if __name__=='__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("input",type=str,
                        help="""Input words for the crossword puzzle. Each line has to be of one of the following forms:

  "<word>: <clue>"                - a word and its corresponding clue
  "#<comment>"                    - comment line; ignored
  "+<Solution>"                   - the solution word or sentence
  "-<info>"                       - info that is printed before clues
  "*<sequence of words>"          - give a sentence to allow asking for single words in the sentence (no colons allowed). Write "§" in front of words that should be asked for (e.g. "I am a §nice §human." allowes the question "I am a nice ???." with answer "HUMAN").

Note: In puzzle and solution, any appearance of Ä, Ö, Ü, and ß is automatically converted to AE, OE, UE, SS, respectively.
""")
    parser.add_argument("--columns","-c",type=int,default=23,
                        help="Maximum number of desired columns for the output puzzle.")
    parser.add_argument("--rows","-r",type=int,default=25,
                        help="Maximum number of desired rows for the output puzzle.")
    parser.add_argument("--title",type=str,default=None,
                        help="Title of the puzzle.")
    parser.add_argument("--subtitle",type=str,default="",
                        help="Subtitle of the puzzle.")
    parser.add_argument("--iterations","-it",type=int,default=100,
                        help="Number of iterations (higher number = possible better puzzle).")
    parser.add_argument("--cycles",action="store_true",
                        help="Searches for cycles. Slows down computation by ~20%.")
    parser.add_argument("--output","-o",type=str,default=None,
                        help="Path to the output file; can be compiled using LaTeX.")
    parser.add_argument("--quiet","-q",action="store_true",
                        help="Does not write anything on the console.")
    parser.add_argument("--english","-en",action="store_true",
                        help="Changes language of output to english (default german).")
    parser.add_argument("--nogray",action="store_true",
                        help="Sets background of cells to white instead of gray.")
    parser.add_argument("--seed", "-s", type=int, default=random.randint(0,4095),
                        help="Specify a seed to make the puzzle creation deterministic.")
    args = parser.parse_args()

        
    if args.output == None:
        ext = "_cwpuzzle_"+hex(args.seed)[2:].upper()+".tex"
        args.output = "".join(args.input.split(".")[:-1])+ext if "." in args.input else args.input+ext

    if args.title == None:
        args.title = "Crossword Puzzle" if args.english else "Kreuzworträtsel"

    with open(args.input, "r") as f:
        lines = f.readlines()

    random.seed(args.seed)
    
    solutions = []
    wordnum = 0 # words without sentences
    wordDict = dict()
    sentenceDict = dict()
    info = []
    for i,line in enumerate(lines):
        # comment line
        if line[0] == "#" or line.replace(" ","").replace("\n","") == "":
            continue
        # solution line
        elif line[0] == "+":
            line = line[2:] if line[1] == " " else line[1:]
            solutions.append(convertWord(line.replace("\n","")))
        # info line
        elif line[0] == "-":
            info.append(line[1:])
        # sentence line
        elif line[0] == "*":
            sentenceWords = [w[1:] for w in line[1:].replace(".","").replace("?","").replace("!","").replace(",","").replace("(","").replace(")","").replace("[","").replace("]","").replace("{","").replace("}","").replace("\"","").replace("\n","").split(" ") if w[0] == "§"]
            #clue = r"``\textit{"+line.replace("§","").replace("\n","")+"}''"
            clue=line[1:].replace("§","").replace("\n","")
            for w in sentenceWords:
                keyword = convertWord(w)
                wordDict[keyword] = clue.replace(w,"???")
                sentenceDict[keyword] = [convertWord(word) for word in sentenceWords]
        # normal "word: clue" line
        elif ": " in line:
            line = line.split(": ")
            wordDict[convertWord(line[0])] = ": ".join(line[1:]).replace("\n","")
            wordnum += 1

                
    t0 = time.time()
    best = crossword()
    random.shuffle(solutions)
    solution = None
    for i in range(args.iterations):
        c = crossword.generate(wordDict,sentenceDict,args.columns,args.rows)
        if not(c.score(args.columns,args.rows) > best.score(args.columns,args.rows)):
            continue
        if len(solutions) == 0:
            best = c
            if not(args.quiet):
                print("Found new one at iteration "+str(i)+"! New score: "+str(best.score())[:5])
        else:
            for sol in solutions:
                if c.setSolution(sol):
                    solution = sol
                    best = c
                    if not(args.quiet):
                        print("Found new one at iteration "+str(i)+"! New score: "+str(best.score())[:5])
                    break
    t1 = time.time()
    c = best
    if len(solutions) > 0 and solution == None:
        raise Exception("Could not produce puzzle with one of the given solutions.")
        
    if not(args.quiet):
        print(c)
        print("scores:     "+str(c.scores()))
        print("score:      "+str(c.score())[:7])
        print("size:       "+str(c.numCols())+"x"+str(c.numRows()))
        print("#words:     "+str(c.numWords()))
        if args.cycles:
            print("#cycles:    "+str(c.numCycles()))
        print("given:      "+str(wordnum)+" words, "+str(len(sentenceDict))+" sentences")
        print("Sol:        "+str(solution))
        print("Seed (hex): "+str(hex(args.seed)[2:].upper()))
        print("Seed (int): "+str(args.seed))
        print("Time:       "+str(t1-t0)[:5] + " s")

    with open(args.output,"w") as f:
        print(latex(c,args.title,args.subtitle,info),file=f)
