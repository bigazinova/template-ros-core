
def solution(obs):
    hsv = cv.cvtColor(obs, cv.COLOR_BGR2HSV)
    lower = np.array([20,100,100])
    upper = np.array([30,255,255])
    mask = cv.inRange(hsv,lower,upper)
    ratio = (cv.countNonZero(mask))/(obs.size/3)
    res = np.round(ratio*100,2)
    print(obs.shape)
    
    if res > 2:
        return [1, 0]
    else:
        return [0, 0]
