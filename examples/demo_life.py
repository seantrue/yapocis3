#!/usr/bin/python 
"""A minimal implementation of Conway's Game of Life.
 
Each cell's survival depends on the number of occupied nearest and
next-nearest neighbours (calculated in Grid::step). A living cell dies
of overcrowding or loneliness if it has more than three or fewer than
two neighbours; a dead cell is brought to life if it has exactly three
neighbours (determined in Cell::setNextState).
    Iain Haslam, June 2005.
""" 
from tkinter import *
root = Tk()
 
class Cell(Label):
    DEAD = 0
    LIVE = 1
    def __init__(self,parent):
        Label.__init__(self,parent,relief="raised",width=2,borderwidth=1)
        self.bind("&lt;Button-1&gt;", self.toggle)
        self.displayState(Cell.DEAD)
 
    def toggle(self,event):
        self.displayState(1-self.state)
 
    def setNextState(self,numNeighbours):
        """Work out whether this cell will be alive at the next iteration.""" 
        if self.state==Cell.LIVE and \
            (numNeighbours>3 or numNeighbours<2):
            self.nextState = Cell.DEAD
        elif self.state==Cell.DEAD and numNeighbours==3:
            self.nextState = Cell.LIVE
        else:
            self.nextState = self.state
 
    def stepToNextState(self):
        self.displayState(self.nextState)
 
    def displayState(self,newstate):
        self.state = newstate
        if self.state==Cell.LIVE:
            self["bg"] = "black" 
        else:
            self["bg"] = "white" 
 
class Grid:
    def __init__(self,parent,sizex,sizey):
        self.sizex = sizex
        self.sizey = sizey
        #numpy.zeros(sizex,sizey) is a better choice,
        #but an additional dependency might be rude...
        self.cells = []
        for a in range(0,self.sizex):
            rowcells = []
            for b in range(0,self.sizey):
                c = Cell(parent)
                c.grid(row=b, column=a)
                rowcells.append(c)
            self.cells.append(rowcells)
        if sizex>10 and sizey>10:
            #Start with a glider
            self.cells[10][9].displayState(Cell.LIVE)
            self.cells[11][10].displayState(Cell.LIVE)
            self.cells[9][11].displayState(Cell.LIVE)
            self.cells[10][11].displayState(Cell.LIVE)
            self.cells[11][11].displayState(Cell.LIVE)
 
    def step(self):
        """Calculate then display the next iteration of the game of life.
 
        This function uses wraparound boundary conditions.
        """ 
        cells = self.cells
        for x in range(0,self.sizex):
            if x==0: x_down = self.sizex-1
            else: x_down = x-1
            if x==self.sizex-1: x_up = 0
            else: x_up = x+1
            for y in range(0,self.sizey):
                if y==0: y_down = self.sizey-1
                else: y_down = y-1
                if y==self.sizey-1: y_up = 0
                else: y_up = y+1
                sum = cells[x_down][y].state + cells[x_up][y].state + \
                    cells[x][y_down].state + cells[x][y_up].state + \
                    cells[x_down][y_down].state + cells[x_up][y_up].state + \
                    cells[x_down][y_up].state + cells[x_up][y_down].state
                cells[x][y].setNextState(sum)
        for row in cells:
            for cell in row:
                cell.stepToNextState()
 
    def clear(self):
        for row in self.cells:
            for cell in row:
                cell.displayState(Cell.DEAD)
 
if __name__ == "__main__":
    frame = Frame(root)
    frame.pack()
    grid = Grid(frame,25,25)
    bottomFrame = Frame(root)
    bottomFrame.pack(side=BOTTOM)
    buttonStep = Button(bottomFrame,text="Step",command=grid.step)
    buttonStep.pack(side=LEFT)
    buttonClear = Button(bottomFrame,text="Clear",command=grid.clear)
    buttonClear.pack(side=LEFT,after=buttonStep)
    root.mainloop()
    
