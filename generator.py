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
        self.letterkeys = set() # same as self.letters.keys()
        # letter_positions["A"] == {(1,2),(4,3)}
        self.letter_positions = dict([ (l,set()) for l in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" ])
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
    def get(self,coord):
        try:
            return self.letters[coord]
        except KeyError:
            return " "
    def isLetter(self,coord,l):
        """Checks whether letters[x,y] == l (more efficient than 'letters[x,y] == l'."""
        if l == " ":
            return not( coord in self.letterkeys )
        else:
            return coord in self.letter_positions[l]
    def getSolution(self):
        return self.solution
    def set(self,x,y,l):
        self.letters[x,y] = l
        self.letterkeys.add( (x,y) )
        self.letter_positions[l].add((x,y))
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
        # if args is None (i.e. test case), then we search for cycles
        if args is not None and args.nocycles:
            return 0
        else:
            return self.graph.numCycles()
    def __le__(self,other):
        # checks if self is subset of other
        return self.wordsHor.items() <= other.wordsHor.items() \
            and self.wordsVer.items() <= other.wordsVer.items()
    def __str__(self):
        n = self.numRows()
        m = self.numCols()
        L = [ [" " for i in range(m)] for i in range(n) ]
        for d in self.letterkeys:
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
        global args
        n = self.numRows()
        m = self.numCols()
        # L[y][x] <-> cell (x,y)
        L = [ ["{}" for i in range(m)] for i in range(n) ]
        # insert letters
        for d in self.letterkeys:
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
                L[n-(d[1]-self.yMin+1)][d[0]-self.xMin] = r"[\cwNumText{"+str(indices[0]+1)+"/"+str(indices[1]+1)+"}{"+str(self.solutionDict[d])+"}][gfo]"+"{"+self.get(d)+"}"
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
        # print crosses in empty boxes surrounded by letters if argument nogray is set
        if args.nogray:
            # iterates over all letters and checks whether coordinate below (col,row)
            # is an empty box surrounded by letters
            for x,y in self.letterkeys:
                if {(x-1,y-1), (x+1,y-1), (x,y-2)}.issubset(self.letterkeys) and not( (x,y-1) in self.letterkeys):
                    L[n-(y-1-self.yMin+1)][x-self.xMin] = "[][/,]{ }"
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
        if args is None or not(args.nocycles):
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
            if coord in self.letterkeys:
                self.intersections.add(coord)
            self.set(coord[0],coord[1],l)
            if args is not None and args.nocycles:
                continue
            self.graph.add_node(coord)
            if k > 0:
                self.graph.add_edge(prev,coord)
        return
    def isAddable(self,word,x,y,hor=True):
        """Checks whether a given word can be added at position x,y"""
        word = word.upper()
        relevantKeys = self.letterkeys
        for k in range(len(word)):
            coords = (x+k,y) if hor else (x,y-k)
            left = (x+k,y+1) if hor else (x+1,y-k)
            right = (x+k,y-1) if hor else (x-1,y-k)
            # checks whether letter would overwrite other letter
            if not(self.isLetter(coords,word[k]) or self.isLetter(coords," ")):
                #raise Exception("Cannot add word "+word+" at position "+str((x,y))+" because of letter "+self.get(coords)+" instead of "+word[k]+" at position "+str(coords)+".")
                return False
            # checks whether neighbouring positions are only letters of intersections words
            if not(self.isLetter(coords,word[k])) and not(self.isLetter(left," ") and self.isLetter(right," ")):
                #raise Exception("Cannot add word "+word+" at position "+str((x,y))+" because it would neighbour other word.")
                return False
            # checks whether position is already taken by two words
            if coords in self.intersections:
                #raise Exception("Cannot add word "+word+" at position "+str((x,y))+" because already two words intersect in letter "+str(word[k])+" at position "+str((x+k,y)))
                return False
            # checks whether letter is not beginning of other word
            wordDict = self.wordsHor if hor else self.wordsVer
            if coords in wordDict.keys():
                #raise Exception("Cannot add word "+word+" at position "+str((x,y))+" as it would intersect with word "+wordDict[coords]+" at position "+str(coords)+".")
                return False
            # checks whether word runs parallel to other word
            if hor and k < len(word):
                if ((x+k,y+1) in relevantKeys and (x+k+1,y+1) in relevantKeys)\
                   or ((x+k,y-1) in relevantKeys and (x+k+1,y-1) in relevantKeys):
                    #raise Exception("Cannot add word "+word+" at position "+str((x,y))+" as it would run parallel to another word.")
                    return False
            elif not(hor) and k < len(word):
                if ((x+1,y-k) in relevantKeys and (x+1,y-k-1) in relevantKeys)\
                   or ((x-1,y-k) in relevantKeys and (x-1,y-k-1) in relevantKeys):
                    #raise Exception("Cannot add word "+word+" at position "+str((x,y))+" as it would run parallel to another word.")
                    return False
        nextCoord = (x+len(word),y) if hor else (x,y-len(word))
        prevCoord = (x-1,y) if hor else (x,y+1)
        # checks whether there is a letter before or after the word
        if nextCoord in relevantKeys or prevCoord in relevantKeys:
            #raise Exception("Cannot add word "+word+" at position "+str((x,y))+" as there is a new letter after its end.")
            return False
        return True
    def isAddable2(self,word,x,y,hor=True):
        """Less efficient but more elegant version of isAddable"""
        word = word.upper()
        relevantKeys = self.letterkeys
        wordDict = self.wordsHor if hor else self.wordsVer
        nextCoord = (x+len(word),y) if hor else (x,y-len(word))
        prevCoord = (x-1,y) if hor else (x,y+1)
        try:
            next(k for k,l in enumerate(word) \
                 if (coords := (x+k,y) if hor else (x,y-k)) \
                 and (left := (x+k,y+1) if hor else (x+1,y-k)) \
                 and (right := (x+k,y-1) if hor else (x-1,y-k)) \
                 and (not(self.isLetter(coords,l) or self.isLetter(coords," ")) \
                      or (not(self.isLetter(coords,l)) and (not(self.isLetter(left," ")) or not(self.isLetter(right," ")))) \
                      or (coords in self.intersections) \
                      or (coords in wordDict.keys()) \
                      or (hor and k < len(word)) and (((x+k,y+1) in relevantKeys and (x+k+1,y+1) in relevantKeys) or ((x+k,y-1) in relevantKeys and (x+k+1,y-1) in relevantKeys)) \
                      or (not(hor) and k < len(word)) and (((x+1,y-k) in relevantKeys and (x+1,y-k-1) in relevantKeys) or ((x-1,y-k) in relevantKeys and (x-1,y-k-1) in relevantKeys)) \
                      ) \
                 )
        except StopIteration:
            return not(nextCoord in relevantKeys or prevCoord in relevantKeys)
        return False
    def whereIsAddable(self,word) -> set:
        word=word.upper()
        if self.numWords() == 0:
            return {(0,0,True)}
            return {(0,0,False), (0,0,True)}
        possibleCoordinates = set()
        for k in range(len(word)):
            l = word[k]
            coordinates = {j for j in self.letterkeys if self.isLetter(j,l)}
            for x,y in coordinates:
                if self.isAddable(word,x-k,y,True):
                    possibleCoordinates.add( (x-k,y,True) )
                elif self.isAddable(word,x,y+k,False):
                    possibleCoordinates.add( (x,y+k,False) )
        return possibleCoordinates
    def score(self,maxSizeX=None,maxSizeY=None):
        # euclidean norm
        return np.sqrt(sum([s**2 for s in self.scores(maxSizeX,maxSizeY)]))
    def scores_weights(numWords,numCycles,numIntersections,numCols,numRows,numWordsHor,numWordsVer):
        # used by scores and scoresIfAdded
        return (
            1/2*numWords*numCycles,
            3*numIntersections,
            numWords**2*1/numCols,
            numWords**2*1/numRows,
            1/4*numWordsHor*numWordsVer
        )
    def scores(self,maxSizeX=None,maxSizeY=None):
        if self.numWords() == 0:
            return (0,0)
        if maxSizeX != None and self.numCols() > maxSizeX:
            return (0,0)
        if maxSizeY != None and self.numRows() > maxSizeY:
            return (0,0)
        # (many cycles, many intersections, few columns, few rows, wordsHor/wordsVer balanced)
        return crossword.scores_weights(self.numWords(),self.numCycles(),self.numIntersections(),self.numCols(),self.numRows(),len(self.wordsHor),len(self.wordsVer))
    def scoreIfAdded(self,word,x,y,hor=True,maxSizeX=None,maxSizeY=None):
        # euclidean norm
        return np.sqrt(sum([s**2 for s in self.scoresIfAdded(word,x,y,hor,maxSizeX,maxSizeY)]))
    def scoresIfAdded(self,word,x,y,hor=True,maxSizeX=None,maxSizeY=None):
        numWords_new = self.numWords()+1
        xMin_new = min(self.xMin,x)
        yMin_new = min(self.yMin,y) if hor else min(self.yMin,y-len(word)+1)
        xMax_new = max(self.xMax,x+len(word)-1) if hor else max(self.xMax,x)
        yMax_new = max(self.yMax,y)
        numCols_new = xMax_new-xMin_new+1
        numRows_new = yMax_new-yMin_new+1
        if maxSizeX != None and numCols_new > maxSizeX:
            return (0,0)
        if maxSizeY != None and numRows_new > maxSizeY:
            return (0,0)
        if self.numWords() == 0:
            return (0,0,1/len(word),1,0)
        numWordsHor_new = len(self.wordsHor)+1 if hor else len(self.wordsHor)
        numWordsVer_new = len(self.wordsVer) if hor else len(self.wordsVer)+1
        numIntersections_new = self.numIntersections()
        numCycles_new = 0
        nodes_added = []
        edges_added = []
        for k,l in enumerate(word):
            coord = (x+k,y) if hor else (x,y-k)
            prev = (x+k-1,y) if hor else (x,y-k+1) # coordinate before coord
            if coord in self.letterkeys:
                numIntersections_new += 1
            if args is not None and args.nocycles:
                continue
            if not(coord in self.graph.nodes):
                nodes_added.append(coord)
                self.graph.add_node(coord)
            if k > 0:
                self.graph.add_edge(prev,coord)
                edges_added.append((prev,coord))
        if args is None or not(args.nocycles):
            numCycles_new = self.numCycles()
            self.graph.remove_edges(edges_added)
            self.graph.remove_nodes(nodes_added)
        return crossword.scores_weights(numWords_new,numCycles_new,numIntersections_new,numCols_new,numRows_new,numWordsHor_new,numWordsVer_new)
    def setSolution(self,solution):
        global alphabet
        global args
        if self.solution != None:
            raise Exception("Crossword already has a solution.")
        alreadyTaken = set()
        solutionDict = dict() # self.solutionDict is not touched until it is clear that the solution is possible
        for i,l in enumerate(solution.upper().replace(" ","")):
            possibleCoords = [coord for coord in self.letterkeys if self.isLetter(coord,l) and not(coord in alreadyTaken)]
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
                scoreDict = dict()
                for coord in coords:
                    s = c.scoreIfAdded(w,coord[0],coord[1],coord[2],maxSizeX,maxSizeY)
                    if s > 0:
                        scoreDict[coord] = s
                coords = list(scoreDict.keys())
                if len(coords) == 0:
                    laidAside.append(w)
                    continue
                sthChanged = True
                if w in sentenceDict.keys():
                    laidAside = [word for word in laidAside if not(word in sentenceDict[w])]
                    words = [word for word in words if not(word in sentenceDict[w])]
                coords.sort(key=lambda c: scoreDict[c])
                chosenCoord = random.choices(coords,range(1,len(coords)+1))[0]
                c.add(w,chosenCoord[0],chosenCoord[1],chosenCoord[2],wordDict[w])
                added.append((w,chosenCoord))
            words = laidAside
            laidAside = []
        #print("added in order: " + str(added))
        return c
    def generate_bf(wordDict,stentenceDict,maxSizeX=None,maxSizeY=None,iterations=None,timeout=None,solutions=None):
        allCWs = crossword.allCrosswords(wordDict,sentenceDict,maxSizeX,maxSizeY,iterations,timeout)
        if solutions is None:
            return max(allCWs,key=lambda x: x.score())
        else:
            allCWsWithSol = [c for c in allCWs if any(c.setSolution(sol) for sol in solutions)]
            if allCWsWithSol == []:
                raise Exception("Could not produce puzzle with one of the given solutions.")
            return max(allCWsWithSol,key=lambda x: x.score())
    def allCrosswords(wordDict,sentenceDict,maxSizeX=None,maxSizeY=None,iterations=None,timeout=None):
        """Returns a list of all crossword puzzles that can be created from wordDict with given sizes."""
        global allCrosswords_iteration
        global allCrosswords_set
        global allCrosswords_start
        allCrosswords_iteration = None
        allCrosswords_set = set() # crosswords get stored in this set
        allCrosswords_start = time.time()
        c = crossword()
        if iterations is not None:
            allCrosswords_iteration = 0
        c.allCrosswordsFromMe(wordDict,sentenceDict,maxSizeX,maxSizeY,iterations,timeout)
        return allCrosswords_set
    def allCrosswordsFromMe(self,wordDict,sentenceDict,maxSizeX,maxSizeY,iterations,timeout):
        """Helper function for crossword.allCrosswords()"""
        global allCrosswords_iteration
        global allCrosswords_set
        global allCrosswords_start
        if iterations is not None and allCrosswords_iteration >= iterations:
            return
        if timeout is not None and time.time()-allCrosswords_start >= timeout:
            return
        nothingAddable = True
        for w in sorted(wordDict.keys(),reverse=True):
            for coord in self.whereIsAddable(w):
                if self.scoreIfAdded(w,coord[0],coord[1],coord[2],maxSizeX,maxSizeY) == 0:
                    continue
                nothingAddable = False
                d = self.copy()
                d.add(w,coord[0],coord[1],coord[2],wordDict[w])
#                if any(d <= c for c in allCrosswords_set):
#                    continue
                wordDict_new = wordDict.copy()
                del wordDict_new[w]
                if w in sentenceDict.keys():
                    for s in set(sentenceDict[w])-{w}:
                        if s in wordDict_new: # s may already be deleted by another sentence
                            del wordDict_new[s]
                d.allCrosswordsFromMe(wordDict_new,sentenceDict,maxSizeX,maxSizeY,iterations,timeout)
        if nothingAddable: # stays True iff no word could be added
            if iterations is not None:
                allCrosswords_iteration += 1
            allCrosswords_set.add(self)

    
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
        return "graph("+str(sorted(list(self.nodes)))+", {"+", ".join([str(tuple(e)) for e in self.edges])+"})"
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
        + "\\begin{document}\n" \
        + ("\\begin{landscape}\n" if args.landscape else "") \
        + printHeader(title,subtitle)+"\n\n\n"+printPuzzleDefs()+"\n\n" \
        + c.latexPuzzle() \
        + "\n\n\\vspace{0.5cm}\n\n" \
        + printSolution(c.solution) \
        + "\n\n" \
        + ("\\end{landscape}\n" if args.landscape else "") \
        + "\\newpage\n" \
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
    if solution is None:
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
    sol="Solution" if args is not None and args.english else "Lösung"
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
    global seed
    if args is not None and args.nogray:
        cell_color = "1"
    else:
        cell_color = ".95"
    return r"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%   Preamble   %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\documentclass["""+("a3paper" if args.a3 else "a4paper")+r""",twoside,11pt]{article}

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
\fancyhead[L]{\scriptsize\color{darkgray} """+hex(seed if args is not None else "").removeprefix("0x").upper()+r"""}

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
args = None # important if __name__ != '__main__'
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
    parser.add_argument("--no-output","-no",action="store_true",
                        help="Does not output a puzzle (only shows infos).")
    parser.add_argument("--columns","-c",type=int,default=None,
                        help="Maximum number of desired columns for the output puzzle.")
    parser.add_argument("--rows","-r",type=int,default=None,
                        help="Maximum number of desired rows for the output puzzle.")
    parser.add_argument("--title",type=str,default=None,
                        help="Title of the puzzle.")
    parser.add_argument("--subtitle",type=str,default="",
                        help="Subtitle of the puzzle.")
    parser.add_argument("--iterations","-it",type=int,default=100,
                        help="Number of iterations (higher number = possible better puzzle).")
    parser.add_argument("--nocycles",action="store_true",
                        help="Does not search for cycles (speeds up computation by ~20%%).")
    parser.add_argument("--output","-o",type=str,default=None,
                        help="Path to the output file; can be compiled using LaTeX.")
    parser.add_argument("--quiet","-q",action="store_true",
                        help="Does not write anything on the console.")
    parser.add_argument("--english","-en",action="store_true",
                        help="Changes language of output to english (default german).")
    parser.add_argument("--nogray",action="store_true",
                        help="Sets background of cells to white instead of gray.")
    parser.add_argument("--seed", "-s", type=int,
                        help="Specify a seed to make the puzzle creation deterministic.")
    parser.add_argument("--hexseed","-hs",type=str,
                        help="Same as --seed but input is interpreted as hexadecimal.")
    parser.add_argument("--bruteforce","-bf",action="store_true",
                        help="Computes puzzle by brute-force search.")
    parser.add_argument("--a3",action="store_true",
                        help="Output is in A3.")
    parser.add_argument("--landscape",action="store_true",
                        help="Output is in landscape.")
    parser.add_argument("--minscore",type=int,
                        help="Computes a puzzle with a score at least the given. Overwrites --iterations. Warning: May not terminate if given number is too high!")
    args = parser.parse_args()


    if (args.seed is not None) and (args.hexseed is not None):
        raise Exception("Please only specify an integer seed OR a hexadecimal seed.")

    if args.hexseed is not None:
        seed = int(args.hexseed,16)
        random.seed()
    elif args.seed is not None:
        seed = args.seed
    else:
        seed = random.randint(0,4095)

    random.seed(seed)
    if not(args.quiet):
        print("Seed was", seed, "(int)")


    if args.columns is None:
        args.columns = 53 if args.a3 and args.landscape else \
            35 if args.a3 and not(args.landscape) else \
            35 if not(args.a3) and args.landscape else \
            22 # if a4 and portrait
    if args.rows is None:
        args.rows = 28 if args.a3 and args.landscape else \
            46 if args.a3 and not(args.landscape) else \
            16 if not(args.a3) and args.landscape else \
            25 # if a4 and portrait

    if args.output is None:
        ext = "_cwpuzzle_"+hex(seed).removeprefix("0x").upper()+".tex"
        args.output = "".join(args.input.split(".")[:-1])+ext if "." in args.input else args.input+ext

    if args.title is None:
        args.title = "Crossword Puzzle" if args.english else "Kreuzworträtsel"

    with open(args.input, "r") as f:
        lines = f.readlines()


    # parse input file
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
            all_words = line[1:].replace(".", " ").replace("?", " ").replace("!", " ").replace(",", " ")\
                .replace("(", " ").replace(")", " ").replace("[", " ").replace("]", " ").replace("{", " ")\
                .replace("}", " ").replace("\"", " ").replace("\n", " ")\
                .split()  # splits on any whitespace and ignores empty strings by default.
            sentenceWords = [w[1:] for w in all_words if len(w) > 1 and w[0] == "§"]
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


    # check whether solutions are possible
    all_letters = [ l for w in wordDict.keys() for l in w ]
    from collections import Counter
    def is_submultiset(l1, l2):
        c1, c2 = Counter(l1), Counter(l2)
        return all(c1[k] <= c2[k] for k in c1)
    if len(solutions) > 0 and not(any(is_submultiset(s.replace(" ",""),all_letters) for s in solutions)):
        s = solutions[0].replace(" ","")
        c1, c2 = Counter(s), Counter(all_letters)
        missing_letters = [ (l,c1[l]-c2[l]) for l in set(s) if c1[l] > c2[l] ]
        raise Exception("None of the given solutions is possible for the given words. Missing letters for first solution:\n"
                        +str(missing_letters))


    t0 = time.time()
    best = crossword()
    random.shuffle(solutions)
    solution = None

    if args.bruteforce:
        c = crossword.generate_bf(wordDict,sentenceDict,args.columns,args.rows,iterations=args.iterations,solutions=solutions)
        solution = c.solution
    else:
        i = 0
        while True:
            i += 1
            if args is not None and not(args.quiet):
                print("Doing iteration Nr. " + str(i),end="\r")
            c = crossword.generate(wordDict,sentenceDict,args.columns,args.rows)
            if args.minscore is None and i >= args.iterations:
                break
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
            if args.minscore is not None and solution is not None and best.score(args.columns,args.rows) >= args.minscore:
                break
        c = best
    
    t1 = time.time()
    if len(solutions) > 0 and solution is None:
        raise Exception("Could not produce puzzle with one of the given solutions.")
        
    if not(args.quiet):
        print(c)
        print("scores:     "+str(c.scores()))
        print("score:      "+str(c.score())[:7])
        print("size:       "+str(c.numCols())+"x"+str(c.numRows()))
        print("#words:     "+str(c.numWords()))
        if not(args.nocycles):
            print("#cycles:    "+str(c.numCycles()))
        print("given:      "+str(wordnum)+" words, "+str(len(sentenceDict))+" sentences")
        print("Sol:        "+str(solution))
        if not(args.bruteforce):
            print("Seed (hex): "+str(hex(seed)[2:].upper()))
            print("Seed (int): "+str(seed))
        print("Time:       "+str(t1-t0)[:5] + " s")
        print("Iterations: "+str(i))

    if not(args.no_output):
        with open(args.output, "w", encoding="utf-8") as f:
            print(latex(c,args.title,args.subtitle,info),file=f)
