import sys
from counsel_therapist import CounselTherapist
from counsel_client import CounselClient


def main():
    """가상 상담 세션 생성
    
    Usage:
        python gen_virtual.py <persona_name>
        
    Example:
        python gen_virtual.py 7_doyoon
    """
    if len(sys.argv) < 2:
        print("사용법: python gen_virtual.py <persona_name>")
        print("예시: python gen_virtual.py 7_doyoon")
        sys.exit(1)
    
    persona_name = sys.argv[1]
    
    # 챗봇 초기화
    therapist = CounselTherapist()
    client = CounselClient(persona_name)
    
    print(f"\n{'='*60}")
    print(f"챗봇 상담 세션 시작 (페르소나: {persona_name})")
    print(f"{'='*60}\n")
    
    # 상담자가 먼저 인사
    therapist_turn = 1
    therapist_message = "안녕하세요. 반가워요"
    print(f"T{therapist_turn}: {therapist_message}\n")
    
    # 20턴 대화 진행
    for turn in range(1, 5):
        # 내담자 응답
        client_message = client.say(therapist_message)
        print(f"C{turn}: {client_message}\n")
        
        # 마지막 턴이면 종료
        if turn == 20:
            break
        
        # 상담자 응답
        therapist_turn += 1
        therapist_message = therapist.say(client_message)
        print(f"T{therapist_turn}: {therapist_message}\n")
    
    print(f"{'='*60}")
    print("상담 세션 종료")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
