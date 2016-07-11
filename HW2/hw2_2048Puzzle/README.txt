Artifical Intellegence Homework 2: Adversarial search (2048 Puzzle) 

Name: Yingtao Xu
UNI: yx2318

PlayerAI Description:

This PlayerAI is designed to move the tiles to get higher score for the 2048 puzzle. 
The algorithms used are minmax algorithm and alpha-beta pruning. The results are reasonable. 
Without pruning, 2048 can hardly be reached, around 5% (1 out of 20 time tests), 1024 can be reached 40%, 512 can be reached 60%, and 256 can be reached 100%.
With pruning, the results are much better. 2048 can be reached for 30% of the tests, 1024 can be reached for 70% and 512 is 100%.

The following are the detialed descrption for the AI.

* There are three classes: GridHelper, Evaluator, and PlayerAI

	* GridHelper is designed to speed up for getting available moves.
		* Only one method, getAbailableMoves, is designed in this class 
		  which is to override the one in Grid class to get faster moves. 

	* Evaluator is the main class for heuristics to get to the optimal algorithm for favorable positions.
		* The total score of one grid is the result after the sum of base score, 
		  the score with consideration of corner, merge, and monotonicity 
		  minus the penality of cross and dead situations. 

		* Each score is obtained by deffirent methods:
			* get_score method: it is to get the total score of one grid descriped above.

			* base method: it is the method to get the base score of the grid which is the sum of 
			  the square values of each cell. This method encorages the grid to merge as much as it can. 

			* corner_boost: it is the method to give more scores for corner situations. This method encorages
			  the larger tiles to move to the corner.

		  	* merge_boost: it is the method to give higher score for the situation where more tiles can be merged.
		  	  This method encorages the tiles to move in the way that more tiles can be merged.

	  	  	* monotonicity_boost: it is the method to give higher score for the situation where the values of the tiles
	  	  	  are all either increasing or decresing along both the left/right and up/down directions.

  	  	  	* cross_penalize: it is the method to give penalty for the situation where the tiles with the same 
  	  	  	  values move to be placed in a diagonal. Take the situationas below an example, 2 and 4 are put in the cross situation so it will get penalty.  

  	  	  		----------
  	  	  		|0 0 0 0 |
  				|0 0 0 0 |
  				|0 2 4 0 |
  				|0 4 2 8 |
  				----------

			* dead_penalize: it is the method to give negative infinity penalty when there is no move at all
	
	* PlayerAI is the class for the move of the tile with the minmax algorithm with and without pruning. 
	  If the member variable ALPHA_BETA_PRUNING is False, it will go with minmax algorithm only while when 
	  ALPHA_BETA_PRUNING is Ture, it will go with alpha-beta pruning.

  	* Notice: sometimes the AI may only move several steps and stop becasue of the time limit. Please change the braching 
  			  factor in the computer's turn in PlayAI to adjust it. 





