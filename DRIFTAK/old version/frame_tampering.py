import cv2
import numpy as np
import os

class StateMask:
    def __init__(self, desiredH = 96, desiredW = 96, save = False, savePath = "./saved-frames") -> None:
        self.desiredHeight = desiredH
        self.desiredWidth = desiredW
        self.save = save
        self.savePath = savePath

    def applyMask(self, state, episode = None, iteration = None):
        """
        It applies the mask to the desired state (96x96x3 array)
        As a result we will obtain the state array which with limited dimensions
        """
        cut = state[0:self.desiredHeight, 0:self.desiredWidth]

        #possibly implement downscaling and some other techniques
        result = cut

        if self.save:
            if not os.path.isdir(self.savePath):
                os.mkdir(self.savePath)
                
            episodename = os.path.join(self.savePath, f"episode-{episode}")

            if not os.path.isdir(episodename):
                os.mkdir(episodename)
                
            filepath = os.path.join(episodename, f"frame-{iteration}.png")
            cv2.imwrite(filepath, result)

        return result
    
    def showOneResult(self, state):
        cv2.imshow("Racing Frame", self.applyMask(state))
        cv2.waitKey()

# print("\n---------------\nState:\n---------------\n", state)
# print("\n---------------\nAction:\n---------------\n", action) 
# print( "\n---------------\nReward:\n---------------\n", reward)
# print( "\n---------------\nNext_state:\n---------------\n", next_state)
# sys.exit()