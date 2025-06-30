from genetic_programming.primitives import primitive_set

def main():
    print("Aruments:")
    for i, name in enumerate(primitive_set.arguments):
        print(f"ARG{i} : {name}")

if __name__ == "__main__":
    main()