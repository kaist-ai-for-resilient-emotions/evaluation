import argparse
import sys

from counsel_therapist import CounselTherapist
from counsel_client import CounselClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="가상 상담 세션을 생성하고 20턴 대화를 출력합니다.",
    )
    parser.add_argument(
        "persona_name",
        help="페르소나 파일 이름 (예: 7_doyoon 또는 personas/7_doyoon.txt)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="gemini-default",
        choices=CounselTherapist.available_models(),
        help="상담자 LLM 모델 (기본값: gemini-default)",
    )
    parser.add_argument(
        "-t",
        "--turns",
        type=int,
        default=20,
        help="내담자 발화 횟수 (기본값: 20)",
    )
    parser.add_argument(
        "--greeting",
        default="안녕하세요. 오늘 어떤 이야기를 나누고 싶으신가요?",
        help="상담자의 첫 인사 문장",
    )
    return parser.parse_args()


def main():
    """가상 상담 세션 생성."""
    args = parse_args()

    if args.turns <= 0:
        print("--turns 값은 1 이상이어야 합니다.")
        sys.exit(1)

    therapist = CounselTherapist.create(args.model)
    client = CounselClient(args.persona_name)

    print(f"\n{'='*60}")
    print(f"가상 상담 세션 시작 (페르소나: {args.persona_name}, 상담자 모델: {args.model})")
    print(f"{'='*60}\n")

    therapist_turn = 1
    therapist_message = args.greeting
    print(f"T{therapist_turn}: {therapist_message}\n")

    for turn in range(1, args.turns + 1):
        client_message = client.say(therapist_message)
        print(f"C{turn}: {client_message}\n")

        if turn == args.turns:
            break

        therapist_turn += 1
        therapist_message = therapist.say(client_message)
        print(f"T{therapist_turn}: {therapist_message}\n")

    print(f"{'='*60}")
    print("상담 세션 종료")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
