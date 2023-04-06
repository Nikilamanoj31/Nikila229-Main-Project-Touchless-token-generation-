import cv2
import mediapipe as mp
import time
import math
 

class handDetector():
    def __init__(self, mode=False, maxHands=2, complexity=1,
                  detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.count = 0
        self.num = 0
 
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.complexity,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
 
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)
 
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img
 
    def findPosition(self, img, handNo=0, draw= True ):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (bbox[0],bbox[1]),
                              (bbox[2]+20, bbox[3]+20), (0, 255, 0), 2)

        return self.lmList, bbox
    
    def findDistance(self, p1, p2, img, draw=True):

        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length,img, [x1, y1, x2, y2, cx, cy]
    
    def fingersUp(self):
        fingers = []

        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers
    
    # def generateToken(self,img):
    #     count=0
    #     j=0
    #     if self.results.multi_hand_landmarks:
    #         if j==0:
    #             count=count+1
    #             j=1
    #             for handLms in self.results.multi_hand_landmarks:
    #                 self.mpDraw.draw_landmarks(img, handLms,
    #                                        self.mpHands.HAND_CONNECTIONS)
    #         else:
    #             j=0
    #         cv2.putText(img,  " Token "+str(count),  (10,50), 
    #                      cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255),  3)
    #         return img, count
    
    def generateToken(self,img,):
        
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if self.num == 0:
                    self.count = self.count+1
                    self.num = 1
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        else:
            self.num = 0
        
        cv2.putText(img, f"Hand Count: {self.count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return img

           
        

  
def main():
    pTime = 0
    cTime = 0
    wcam = 640
    hcam = 480
    frameR = 100 # Frame Reduction
    smoothening = 7
    cap = cv2.VideoCapture(0)
    cap.set(3,wcam)
    cap.set(4,hcam)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        Count = detector.generateToken(img)
        #if len(lmList) != 0:
            #print(lmList[4])
        #print(lmList)
 
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
 
        cv2.putText(img, str(int(fps)), (450,50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

 
        cv2.imshow("Image", img)
        cv2.moveWindow("Image",500,50)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()