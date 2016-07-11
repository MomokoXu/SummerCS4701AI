#!/usr/bin/env python
#coding:utf-8

from random import randint
from BaseAI import BaseAI
from Grid import Grid

class PlayerAI(BaseAI):
	def getMove(self, grid):
		return self.DFS(grid)

	def DFS(self, grid):
		stack = list()
		stack.append(grid)
		visited = set()
		maxValue = 0
		paths = [[]]
		depths = [0]
		maxPath = []
		while stack:
			newGrid = stack.pop()
			depth = depths.pop()
			pathTemp = paths.pop()

			if newGrid in visited:
				continue
			visited.add(newGrid)
			if depth > 5:
				continue

			availableMoves = newGrid.getAvailableMoves()

			for m in availableMoves:
				if m == 0 and len(availableMoves) != 1:
					continue
				nextGrid = newGrid.clone()
				nextGrid.move(m)
				insertRandonTile(nextGrid)
				depths.append(depth + 1)
				stack.append(nextGrid)
				path = list(pathTemp)
				path.append(m)
				paths.append(path)
				maxGrid = newGrid.clone()
				totalValue = 0
				for pa in paths:
					for p in pa:
						maxGrid.move(p)
						for x in xrange(maxGrid.size):
							for y in xrange(maxGrid.size):
								totalValue += maxGrid.getCellValue((x, y))
						if maxValue < totalValue:
							maxValue = totalValue
							maxPath = pa
		return maxPath[0]


Possibility = 0.9
PossibleNewTileValue = [2, 4]

def getNewTileValue():
		if randint(0,99) < 100 * Possibility: 
			return PossibleNewTileValue[0] 
		else: 
			return PossibleNewTileValue[1];


def insertRandonTile(grid):
		tileValue = getNewTileValue()
		cells = grid.getAvailableCells()
		cell = cells[randint(0, len(cells) - 1)]
		grid.setCellValue(cell, tileValue)











