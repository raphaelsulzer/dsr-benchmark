def set_paths(DATA_DIR,CPP_DIR):
    with open("./DATA_DIR.txt", "w") as f:
        f.write(DATA_DIR)
    with open("./CPP_DIR.txt", "w") as f:
        f.write(CPP_DIR)


