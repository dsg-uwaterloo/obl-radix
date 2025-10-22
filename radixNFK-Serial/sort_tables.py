import sys

DATA_LENGTH = 12

def is_blank_line(line):
    return line.strip() == ""

def read_table(lines, count):
    table = []
    while len(table) < count:
        line = next(lines)
        if is_blank_line(line):
            continue
        key, *payload = line.strip('\n').split(' ', 1)
        payload = payload[0] if payload else ""
        payload = payload[:DATA_LENGTH-1] 
        table.append((int(key), payload))
    return table

def main(input_path, output_path):
    with open(input_path, 'r') as f:
        lines = iter(f.readlines())

    # Skip blank lines to find header
    for line in lines:
        if not is_blank_line(line):
            n0n1 = line.strip()
            break
    n0, n1 = map(int, n0n1.split())

    table0 = read_table(lines, n0)
    table1 = read_table(lines, n1)

    # Sort tables by key
    table0.sort(key=lambda x: x[0])
    table1.sort(key=lambda x: x[0])

    with open(output_path, 'w') as f:
        f.write(f"{n0} {n1}\n\n")
        for key, payload in table0:
            payload = payload[:DATA_LENGTH-1]  
            f.write(f"{key} {payload}\n")
        f.write("\n")
        for key, payload in table1:
            payload = payload[:DATA_LENGTH-1] 
            f.write(f"{key} {payload}\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python sort_tables.py input.txt output.txt")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
