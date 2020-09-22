#나만의 스타크래프트 만들기 실습 코드

1. 2020-08-21
- 구현 내용
    - racepack으로 각 종족별 모듈 작성
    - Protoss 코드 작성
    - 파일런생성->게이트웨이생성->질럿생성->공격
    
- 미구현 내용
    - 프로브 추가 생성 -> 로직이 꼬임
    - idle worker 선택 후 미네랄 채취 -> 미네랄 위치를 하드코딩 해야 하는지?
    - 파일런 및 게이트웨이 추가 -> 위치 지정을 하드코딩 해야하는지

2. 2020-08-31
- 구현 내용
    - probe, zealot 생산
    - pylon, gateway 건설
        - 건설시 pylon/nexus위치 기준으로 범위 지정하여 생산
    - 건물 생산 점수 및 do nothing panalty 부여
        - panalty 부과시 학습이 더 빨리 진행되는 것이 체감됨

3. 2020-09-07
- 구현 내용
    - assimilator, cybernetics core, forge 건설 추가
    - stalker 생산 추가
    - sparce reward 변경
- 미구현 내용
    - worker들이 assimilator에 제대로 배속되지 않음
    - 건물 위치가 계속 찾을 수 없음으로 나오는 경우가 많음
    
4. 2020-09-15 Final
- 구현 내용
    - ProtossDQN.py 실행 필요
    - stargate, roboticsfacility, roboticsbay 건설 추가
    - mothershipcore, mothership, colossus, immortal, tempest, voidray 생산 추가
    - 앞마당 멀티 1개소 
- 결과
    - 별도의 트레이닝 과정이 없이도 easy모드는 쉽게 이김
    - 난이도를 높여 학습중

Lessons Learned
- 
1. pylon을 너무 많이 지음 --> 여유분을 체크해서 제한할 필요 있음
 - 이유는 다른 건물들 또는 유닛 생산에는 제약이 많아 no_op이 리턴되지만 파일런은 저비용에 제약이 없어 액션의 성공확률이 높으므로 당연히 파일런을 많이 짓는 쪽으로 학습됨
 - 파일런을 너무 안짓는 경우 고난이도 미션에서 생산량을 food가 받쳐주지 못하는 경우가 발생함
2. 멀티를 잡을 때 위치의 경우 vespene gas의 위치를 기준으로 하면 편리함
3. 난이도가 올라갈수록 적이 뽑는 유닛의 종류가 많아 모든 경우의 수를 체크 해 줘야 함
4. 승패에 영향을 주는 요소들을 최대한 state input에 표기해 줘야 학습이 잘 됨
5. state 및 action의 수가 상대적으로 적고, 네트워크가 비교적 단순하기 때문에 hidden layer를 지나치게 많이 잡으면 학습이 더딘 느낌임
