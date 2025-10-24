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
        "--client-model",
        default="default",
        choices=CounselClient.available_models(),
        help="내담자 LLM 모델 (fast=claude-haiku-4-5, default=claude-sonnet-4-5)",
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
        default="안녕하세요. 오늘 어떤 이야기를 나누고 싶으신가요? 편하게 먼저 이야기해 주세요.",
        help="상담자가 내담자에게 전달하는 첫 메시지. 빈 문자열이면 내담자가 스스로 대화를 시작.",
    )
    return parser.parse_args()


def main():
    """가상 상담 세션 생성."""
    args = parse_args()

    if args.turns <= 0:
        print("--turns 값은 1 이상이어야 합니다.")
        sys.exit(1)

    therapist = CounselTherapist.create(args.model)
    client = CounselClient(args.persona_name, model_alias=args.client_model)

    print(f"\n{'='*60}")
    print(
        "가상 상담 세션 시작 "
        f"(페르소나: {args.persona_name}, 상담자 모델: {args.model}, 내담자 모델: {args.client_model} -> {client.model_name})"
    )
    print(f"{'='*60}\n")

    therapist_turn = 0
    therapist_message = (args.greeting or "").strip()
    if therapist_message:
        print(f"(Therapist cue) {therapist_message}\n")
    else:
        therapist_message = (
            "상담자는 아직 말을 꺼내지 않았어. 내담자 스스로 상담을 시작하기 위한 인사와 고민 소개를 해줘."
        )

    for turn in range(1, args.turns + 1):
        client_message = client.say(therapist_message)
        print(f"C{turn}: {client_message}\n")

        if client.has_ended:
            print("(Client end signal detected)\n")
            break

        therapist_turn += 1
        therapist_message = therapist.say(client_message)
        print(f"T{therapist_turn}: {therapist_message}\n")

    print(f"{'='*60}")
    print("상담 세션 종료")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
