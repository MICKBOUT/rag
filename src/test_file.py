import time


def fireball() -> str:
    time.sleep(0.1)
    return "Fireball cast!"


def nested_function_outide() -> None:
    def nested_function_inside(i: int) -> None:
        print(i)
    for i in range(10):
        nested_function_inside(i)


class ClassExemple:
    def basic_class_methode(self) -> int:
        return 1 + 1


def main() -> None:
    fireball()


if __name__ == "__main__":
    main()
