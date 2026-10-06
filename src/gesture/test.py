# 캠 연결하기
import cv2
import mediapipe as mp
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
# 인식 가능한 11가지 동작
gesture = {
    0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok',
}
# 가위바위보 동작 만들기
rsp = {
    0: 'rock', 5: 'paper', 9:'scissors'
}
# 동작 인식 모델 만들기 (knn모델)
file = np.genfromtxt('./data/gesture_train.csv', delimiter = ',')
X = file[:,:-1].astype(np.float32)
y = file[:,-1].astype(np.float32)
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X,y)
# mediapipe 사용하기
# 손찾디 관련 기능 불러오기
mp_hands = mp.solutions.hands
#손 그려주는 기능 불러오기
mp_drawing = mp.solutions.drawing_utils
# 손 찾기 관연 세부 설정
hands = mp_hands.Hands(
    max_num_hands = 2,# 탐지할 최대 손의 객수
    min_detection_confidence = 0.5, # 표시할 손의 최소 정확도
    min_tracking_confidence = 0.5 # 표시할 관잘의 최소 정확도

)

video = cv2.VideoCapture(0)
while video.isOpened():
    ret, img = video.read()
    img = cv2.flip(img,1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img) # 손 탐지하기

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if not ret:
        break
    # 찾은 손 표시하기
    if result.multi_hand_landmarks is not None:
        rsp_result = []
        #d이미지 손 표현하기
       # print(result.multi_hand_landmarks)
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21,3)) # 21개 관절, xyz값 저장할 배열 생성
            # enumerate = for문의 순서 표현
            for j,lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y,lm.z]
                # 연결할 관절 번호 가져오기
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:]
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:]
            v = v2-v1 # 뼈의 값 (x,y,z좌표값 > 백터값)
            # 유클리디안 길이로 변롼(피타고라스)
            # 뼈의 값(직선 값)
            v = v/np.linalg.norm(v, axis = 1)[:, np.newaxis]
            # 뼈의 값으로 뼈사이의 각도 구하기, 변화값이 큰 15개
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))
            # radian 각도를 degree각도로 변경하기
            angle = np.degrees(angle)
            # 구한 각도를 knn모델에 예측시키기
            # 학습을 위해서 타입 변경 (2차원 array)
            X_pred = np.array([angle],dtype = np.float32)
            results = knn.predict(X_pred)
            idx = int(results)
            # 인식된 제스쳐 표현하기
            img_x = img.shape[1]
            img_y = img.shape[0]
            hand_x = res.landmark[0].x
            hand_y = res.landmark[0].y
            #cv2.putText(img, text = gesture[idx].upper(),
             #          org = ( int(hand_x * img_x), int(hand_y * img_y )),
              #         fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2
               #        )

            if idx in rsp.keys(): # 0,5,9 번 일때만 실행
                cv2.putText(img, text = rsp[idx].upper(),
                         org = ( int(hand_x * img_x), int(hand_y * img_y )),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2
                     )
                rsp_result.append({
                     'rsp': rsp[idx],
                     'org': ( int(hand_x * img_x), int(hand_y * img_y )+20),
                })
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
            # 누기 이겼는지 판단하기
            if len(rsp_result) == 2:
                winner = None
                text = ''
                if rsp_result[0]['rsp'] == 'rock':
                    if rsp_result[1]['rsp'] == 'rock': text = 'Tie';
                    elif rsp_result[1]['rsp'] =='paper':text = 'Paper win'; winner =1;
                    elif rsp_result[1]['rsp'] == 'scissors':text ='Rock win'; winner = 0;
                elif rsp_result[0]['rsp'] == 'paper':
                    if rsp_result[1]['rsp'] == 'paper': text = 'Tie';
                    elif rsp_result[1]['rsp'] =='rock':text = 'Paper win'; winner =0;
                    elif rsp_result[1]['rsp'] == 'scissors':text ='Scissors win'; winner = 1;
                elif rsp_result[0]['rsp'] == 'scissors':
                    if rsp_result[1]['rsp'] == 'scissors': text = 'Tie';
                    elif rsp_result[1]['rsp'] =='rock':text = 'Rock win'; winner =1;
                    elif rsp_result[1]['rsp'] == 'paper':text ='Scissors win'; winner = 0;
                cv2.putText(img, text = text,
                            # 이미지 중앙에 결과 표시
                           org =( int(img_x/3),100),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=3
                           )
    k = cv2.waitKey(30)
    if k==49:
        break
    cv2.imshow('hand', img)
video.release()
cv2.destroyAllWindows()
