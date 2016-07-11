#!/usr/bin/env python
#coding:utf-8

from random import randint
from BaseAI import BaseAI
import time
import operator

INF = 9999999999999999
NEG_INF = -INF

class GridHelper():
    @staticmethod
    def getAvailableMoves(grid):
        moves = []
        up = False
        down = False
        left = False
        right = False
        mat = grid.map
        for i in range(4):
            for j in range(4):
                cur = grid.map[i][j]
                if cur != 0:
                    if not left and j > 0 and (grid.map[i][j - 1] == 0 or grid.map[i][j - 1] == cur):
                        left = True
                    if not right and j < 3 and (grid.map[i][j + 1] == 0 or grid.map[i][j + 1] == cur):
                        right = True
                    if not up and i > 0 and (grid.map[i - 1][j] == 0 or grid.map[i - 1][j] == cur):
                        up = True
                    if not down and i < 3 and (grid.map[i + 1][j] == 0 or grid.map[i + 1][j] == cur):
                        down = True
        if down:
            moves.append(1)
        if left:
            moves.append(2)
        if right:
            moves.append(3)
        if up:
            moves.append(0)
        return moves

class Evaluator():
    def get_score(self, grid):
        return self.base(grid) + self.corner_boost(grid) + self.merge_boost(grid) + self.monotonicity_boost(grid) - self.cross_penalize(grid) - self.dead_penalize(grid)

    def base(self, grid):
        score = 0
        coef = 600
        for i in range(4):
            for j in range(4):
                score += grid.map[i][j] * grid.map[i][j] 
        return score * coef

    def corner_boost(self, grid):
        score = 0
        coef = 500
        for i in range(4):
            for j in range(4):
                if (i == 0 and j == 0) or (i == 0 and j == 3) or (i == 3 and j == 0) or (i == 3 and j == 3):
                    score += coef * grid.map[i][j]
        return score

    def merge_boost(self, grid):
        score1 = 1
        score2 = 1
        coef = 100
        for i in range(3):
            cnt = 0
            max_val = 1
            for j in range(4):
                if grid.map[i][j] != 0 and grid.map[i][j] == grid.map[i + 1][j]:
                    cnt += 1
                    if grid.map[i][j] > max_val:
                        max_val = grid.map[i][j]
            if cnt > 2:
                score1 = 500000000
            elif cnt == 2:
                score1 = (2 ** cnt) * max_val * 1000
            else:
                score1 = (2 ** cnt) * max_val * coef

        for j in range(3):
            cnt = 0
            max_val = 1
            for i in range(4):
                if grid.map[i][j] != 0 and grid.map[i][j + 1] == grid.map[i][j]:
                    cnt += 1
                    if grid.map[i][j] > max_val:
                        max_val = grid.map[i][j]
            if cnt > 2:
                score2 = 500000000
            elif cnt == 2:
                score2 = (2 ** cnt) * max_val * 1000
            elif cnt == 1:
                score2 = (2 ** cnt) * max_val * coef
        if score1 == 1 and score2 == 1:
            return -500000
        else:
            return max(score1, score2) 

    def monotonicity_boost(self, grid):
        coef = 200

        horizontal_left_to_right = 0
        for i in range(4):
            tmp_horizontal_cnt = 1
            tmp_power = 1
            for j in range(3):
                if grid.map[i][j] < grid.map[i][j + 1]:
                    tmp_power *= 2
                    tmp_horizontal_cnt += (2 ** tmp_power) * grid.map[i][j + 1]
                elif grid.map[i][j] > grid.map[i][j + 1]:
                    tmp_power *= 2
                    tmp_horizontal_cnt -= (2 ** tmp_power) * grid.map[i][j] * grid.map[i][j +1]
                    tmp_power = 1
            horizontal_left_to_right += tmp_horizontal_cnt * coef

        horizontal_right_to_left = 0
        for i in range(4):
            tmp_horizontal_cnt = 1
            tmp_power = 1
            for j in reversed(range(3)):
                if grid.map[i][j] > grid.map[i][j + 1]:
                    tmp_power *= 2
                    tmp_horizontal_cnt += (2 ** tmp_power) * grid.map[i][j]
                elif grid.map[i][j] < grid.map[i][j + 1]:
                    tmp_power *= 2
                    tmp_horizontal_cnt -= (2 ** tmp_power) * grid.map[i][j + 1] * grid.map[i][j]
                    tmp_power = 1
            horizontal_right_to_left += tmp_horizontal_cnt * coef

        vertical_up_to_down = 0
        for j in range(4):
            tmp_vertical_cnt = 1
            tmp_power = 1
            for i in range(3):
                if grid.map[i][j] < grid.map[i + 1][j]:
                    tmp_power *= 2
                    tmp_horizontal_cnt += (2 ** tmp_power) * grid.map[i + 1][j]
                elif grid.map[i][j] > grid.map[i + 1][j]:
                    tmp_power *= 2
                    tmp_horizontal_cnt -= (2 ** tmp_power) * grid.map[i][j] * grid.map[i + 1][j]
                    tmp_power = 1
            vertical_up_to_down += tmp_vertical_cnt * coef

        vertical_down_to_up = 0
        for j in range(4):
            tmp_vertical_cnt = 1
            tmp_power = 1
            for i in reversed(range(3)):
                if grid.map[i][j] > grid.map[i + 1][j]:
                    tmp_power *= 2
                    tmp_horizontal_cnt += (2 ** tmp_power) * grid.map[i][j]
                elif grid.map[i][j] < grid.map[i + 1][j]:
                    tmp_power *= 2
                    tmp_horizontal_cnt -= (2 ** tmp_power) * grid.map[i + 1][j] * grid.map[i][j]
                    tmp_power = 1
            vertical_down_to_up += tmp_vertical_cnt * coef


        return max(horizontal_left_to_right, horizontal_right_to_left) + max(vertical_down_to_up, vertical_up_to_down)

    def cross_penalize(self, grid):
        directions = [[-1, -1], [1, -1], [-1, 1], [1, 1]]
        penalty = 0
        coef = 500
        for i in range(4):
            for j in range(4):
                for d in directions:
                    ix = i + d[0]
                    iy = j + d[1]
                    if not grid.crossBound((ix, iy)) and grid.map[i][j] == grid.map[ix][iy]:
                        penalty += coef * grid.map[i][j]
        return penalty

    def dead_penalize(self, grid):
        for i in range(4):
            for j in range(4):
                if grid.map[i][j] == 0:
                    return 0
        return -NEG_INF

class PlayerAI(BaseAI):
    SEARCH_TIME = 1
    ALPHA_BETA_PRUNING = False

    def __init__(self):
        self.eval = Evaluator()

    def getMove(self, grid):
        start_time = time.time()
        depth = 1
        opt_move = -1
        opt_score = NEG_INF
        SEARCH_LIMIT = 8 if self.ALPHA_BETA_PRUNING else 7

        while True and depth < SEARCH_LIMIT:
            tmp_move, tmp_score = self.search(grid, depth, 0, NEG_INF, INF, self.ALPHA_BETA_PRUNING)
            if tmp_score > opt_score:
                opt_score = tmp_score
                opt_move =tmp_move
            if time.time() - start_time > self.SEARCH_TIME + 0.3:
                break
            else:
                depth += 1
        return opt_move

    def search(self, grid, limit, depth, alpha, beta, prune):
        if depth == limit:
            return (-1, self.eval.get_score(grid))

        # player's turn
        if depth % 2 == 0:
            opt_score = NEG_INF
            opt_move = -1
            for move in GridHelper.getAvailableMoves(grid):
                # never move up unless it has to
                if move == 0 and opt_score != NEG_INF:
                    continue
                cur_grid = grid.clone()
                cur_grid.move(move)
                _, tmp_score = self.search(cur_grid, limit, depth + 1, opt_score, beta, prune)
                if tmp_score > opt_score:
                    opt_score = tmp_score
                    opt_move = move
                if prune and opt_score > beta:
                    opt_score = beta
                    break
            #print opt_score
            return (opt_move, opt_score)
        # computer's turn
        else:
            tiles_to_insert = []

            # used to get candidates 
            row = [False] * 4
            col = [False] * 4
            for i in range(4):
                for j in range(4):
                    if grid.map[i][j] == 0 and (not row[i] or not col[j]):
                        tiles_to_insert.append((i, j))
                        row[i] = True
                        col[j] = True

            new_grids = []
            candidate_scores = []
            for tile in tiles_to_insert:
                cur_grid = grid.clone()
                cur_grid.insertTile(tile, 2 if randint(0, 99) < 90 else 4)
                candidate_scores.append((cur_grid, self.eval.get_score(cur_grid)))

            # pick worst conditions 
            candidate_scores = sorted(candidate_scores, key = operator.itemgetter(1))
            candidates = [g for (g, s) in candidate_scores]
            candidates = candidates[:4] # change to smaller number to aviod AI stop working due to time limit.

            opt_score = INF
            for cur_grid in candidates:
                _, tmp_score = self.search(cur_grid, limit, depth + 1, alpha, opt_score, prune)
                if tmp_score < opt_score:
                    opt_score = tmp_score
                if prune and opt_score < alpha:
                    opt_score = alpha
                    break
            return (1, opt_score)