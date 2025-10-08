import numpy as np

def seeded_matrix(seed, size, lower, upper):
    np.random.seed(seed)
    return np.random.random((size, size)) * (upper - lower) + lower

def main():
    seeds = [123, 1234, 12345, 123456, 1234567]
    lower = [0, 1, 2, 8, 32]
    upper = [1, 2, 8, 32, 256]

    for i in range(len(seeds)):
        for size in [10**j for j in range(1, 3)]: # 10, 100
            matrix = seeded_matrix(seeds[i], size, lower[i], upper[i])
            
            # save the size first
            with open(f"data/matrix_{size}_{i}.txt", "w") as f:
                f.write(f"{size}\n")
                np.savetxt(f, matrix, fmt="%.6f")

            print(f"Generated matrix_{size}_{i}.txt")


if __name__ == "__main__":
    main()