import numpy as np
import cv2
import chess
import chess.uci

calibrated = False
coords = []

def calBoard(event,x,y,flags,param):
    global coords
    global calibrated
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
     
    elif event == cv2.EVENT_LBUTTONUP:
        if len(coords) < 4:
            coords.append([x, y])
            print('Point' + str(len(coords)) + 'calibrated')
        else:
            calibrated = True


def diffImg(t1, t2):
    d1 = cv2.absdiff(t2, t1)
    return d1

def warpTransform(frame, x1, y1, x2, y2):
    M = cv2.getPerspectiveTransform(pts1,pts2)
    frame3 = cv2.warpPerspective(frame,M,(750,400))
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
 
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
 
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
 
    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
 
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = maxWidth
 
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
	[0, 0],
	[maxWidth - 1, 0],
	[maxWidth - 1, maxHeight - 1],
	[0, maxHeight - 1]], dtype = "float32")
 
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
    # return the warped image
    return warped

def identifyOutliers(data, m=2):
    newData = np.zeros((8,8))
    for i in range(8):
        for j in range(8):
            if(data[i][j] - np.mean(data)) > m * np.std(data):
                newData[i][j] = 1
    return newData

def findActiveSquares(image):
    squareFullness = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            squareStartX = i*(np.size(image, 1)/8)
            squareStartY = j*(np.size(image, 0)/8)
            for k in range(squareStartX, squareStartX+(np.size(image, 1)/8)):
                for l in range(squareStartY, squareStartY+(np.size(image, 0)/8)):
                    if image[k][l] == 255:
                        squareFullness[i][j] = squareFullness[i][j] + 1
    
    squareBinary = np.zeros((8,8))
    return(identifyOutliers(squareFullness))
    
def findPossibleMoves(boardMatrix):
    positions = []
    moves = []
    for i in range(8):
        for j in range(8):
            if boardMatrix[i][j] == 1:
                positions.append(chr(7-j+97)+str(i+1))
    for k in range(len(positions)):
        for l in range(len(positions)):
            if k != l:
                moves.append(positions[k]+positions[l])
    return moves

def findLegalMoves(moves, board):
    legalMoves = []
    for i in range(len(moves)):
        if chess.Move.from_uci(moves[i]) in board.legal_moves:
            legalMoves.append(moves[i])
    return legalMoves
            
    
cap = cv2.VideoCapture(0)

while calibrated == False:
    
    image = cap.read()[1]
    cv2.imshow( "calBoard", image)
    cv2.setMouseCallback('calBoard',calBoard)
    buttonPressed = cv2.waitKey(30) &0xff
    

board = chess.Board()
boardPts = np.array(coords, dtype = "float32")
t_minus = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
t = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
t_plus = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
frame = t_plus
t_plus = four_point_transform(t_plus, boardPts)
t_minus = four_point_transform(t_minus, boardPts)
t = four_point_transform(t, boardPts)


old_image = t
new_image = t
squareSize = 0

engine = chess.uci.popen_engine("C:\stockfish-6-win\Windows\stockfish-6-32.exe")
engine.uci()

while(True):
        
    cv2.imshow( "differences", diffImg(t_minus, t_plus))
    cv2.imshow( "raw", t )
    
    t_minus = t
    t = t_plus
    t_plus = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
    frame = t_plus
    t_plus = four_point_transform(t_plus, boardPts)
    squareSize = np.size(t_plus, 1)/8

    buttonPressed = cv2.waitKey(30) &0xff
    if buttonPressed == ord('q'):
        break
    if buttonPressed == ord('c'):
        old_image = t
        print('CAPTURING FIRST IMAGE')
    if buttonPressed == ord('v'):
        new_image = t
        print('CAPTURING SECOND IMAGE')
        print('GENERATING DIFFERENCE')
        differencedImage = diffImg(old_image, new_image)
        thresh = 30
        differencedImage = cv2.threshold(differencedImage, thresh, 255, cv2.THRESH_BINARY)[1]
        activeSquares = findActiveSquares(differencedImage)
        possibleMoves = findPossibleMoves(activeSquares)
        print possibleMoves
        legalMoves = findLegalMoves(possibleMoves, board)
        print legalMoves
        if len(legalMoves) == 1:
            board.push(chess.Move.from_uci(legalMoves[0]))
            engine.position(board)
            bestMove = engine.go(movetime=2000)[0]
            print bestMove
            board.push(bestMove)
        else:
            print ("MOVE AMBIGUOS")
  
        for i in range(1, 8):
            cv2.line(differencedImage, (i*squareSize, 0), (i*squareSize, 700), (255,0,0), 3)
        for j in range(1, 8):
            cv2.line(differencedImage, (0, j*squareSize), (700, j*squareSize), (255,0,0), 3)
        cv2.imshow( "actualDifferences", differencedImage)
        

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

