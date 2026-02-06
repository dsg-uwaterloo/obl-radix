import sys

def swap_tables(input_path, output_path):
    with open(input_path, "r") as f:
        lines = [line.rstrip("\n") for line in f]

    # Parse n0 and n1
    try:
        n0, n1 = map(int, lines[0].split())
    except Exception:
        print("Error: First line must contain two integers: n0 n1")
        sys.exit(1)

    idx = 1

    # Expect blank line
    if lines[idx] != "":
        print("Error: Expected an empty line after n0 n1")
        sys.exit(1)
    idx += 1

    # Read table 0 (n0 rows)
    table0 = lines[idx : idx + n0]
    idx += n0

    # Expect another blank line
    if lines[idx] != "":
        print("Error: Expected an empty line after table 0")
        sys.exit(1)
    idx += 1

    # Read table 1 (n1 rows)
    table1 = lines[idx : idx + n1]

    # Write output with swapped tables
    with open(output_path, "w") as out:
        out.write(f"{n1} {n0}\n\n")
        for row in table1:
            out.write(row + "\n")
        out.write("\n")
        for row in table0:
            out.write(row + "\n")

    print(f"Swapped tables written to {output_path}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python swap_tables.py <input.txt> <output.txt>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    swap_tables(input_path, output_path)

if __name__ == "__main__":
    main()
